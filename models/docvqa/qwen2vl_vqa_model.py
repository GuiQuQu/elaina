import gc
from typing import List
import torch
from torch import nn
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLConfig,
    Qwen2Tokenizer,
    GenerationConfig,
)

from peft import LoraConfig, get_peft_model

from models.docvqa.qwen2vl.qwen2vl_lora import (
    find_all_linear_modules,
    patch_target_modules,
)

from utils.register import Register

from logger import logger


def get_torch_dtype(model_dtype: str):
    if model_dtype == "bf16":
        return torch.bfloat16
    elif model_dtype == "fp16":
        return torch.float16
    elif model_dtype == "fp32":
        return torch.float32
    else:
        raise ValueError(f"model_dtype {model_dtype} is not supported")


def _freeze_params(module):
    for param in module.parameters():
        param.requires_grad = False


@Register(name="qwen2vl_vqa_model")
class Qwen2VLVQAModel(nn.Module):
    def __init__(
        self,
        model_path,
        warp_qwen2vl_lora: int = 0,
        generation_config=None,
        freeze_vision_model=True,
        freeze_llm_model=False,
        model_dtype="bf16",
    ) -> None:
        super().__init__()
        config = Qwen2VLConfig.from_pretrained(model_path)
        config.use_cache = False
        self.generation_config = GenerationConfig.from_pretrained(model_path)
        if generation_config is not None:
            for key, value in generation_config.items():
                setattr(self.generation_config, key, value)
        logger.info(f"[{self.__class__.__name__}] generation_config: {self.generation_config}")
        self.model: Qwen2VLForConditionalGeneration = (
            Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                config=config,
                torch_dtype=get_torch_dtype(model_dtype),
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2",
            )
        )

        self.tokenizer: Qwen2Tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "left"

        if freeze_vision_model:
            self.model.visual = self.model.visual.eval()
            _freeze_params(self.model.visual)
        if freeze_llm_model:
            self.model.model = self.model.model.eval()
            _freeze_params(self.model.model)

        if warp_qwen2vl_lora:
            self.warp_qwen2vl_lora(
                r=warp_qwen2vl_lora, lora_alpha=2 * warp_qwen2vl_lora
            )

        self.count_parameters()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def count_parameters(self):
        print(self)
        traing_parms = sum(p.numel() for p in self.parameters() if p.requires_grad)
        all_parms = sum(p.numel() for p in self.parameters())
        logger.info(
            f"training parameters: {traing_parms / 10**6} M, all parameters: {all_parms / 10**6} M"
        )

    def forward(
        self,
        pixel_values,
        input_ids,
        attention_mask,
        image_grid_thw,
        labels=None,
    ):

        device = self.model.device
        dtype = self.model.dtype
        loss = None
        pixel_values = pixel_values.to(device=device, dtype=dtype)
        input_ids = input_ids.to(device=device)
        attention_mask = attention_mask.to(device=device)
        image_grid_thw = image_grid_thw.to(device=device)
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )
        loss = outputs.loss
        return loss, outputs

    def generate(self, *arg, **kwargs):
        return self.model.generate(*arg, **kwargs)

    def inference_forward(
        self,
        test_pixel_values,
        test_input_ids,
        test_attention_mask,
        test_image_grid_thw=None,
        test_video_grid_thw=None,
        max_new_tokens=128,
    ):
        return None, self.batch_chat(
            pixel_values=test_pixel_values,
            input_ids=test_input_ids,
            attention_mask=test_attention_mask,
            image_grid_thw=test_image_grid_thw,
            video_grid_thw=test_video_grid_thw,
            return_generation_ids=False,
            max_new_tokens=max_new_tokens,
        )

    def batch_chat(
        self,
        pixel_values,
        input_ids,
        attention_mask,
        image_grid_thw=None,
        video_grid_thw=None,
        return_generation_ids=False,
        max_new_tokens=128,
    ) -> List[str] | str:
        device = self.model.device
        dtype = self.model.dtype
        pixel_values = pixel_values.to(device=device, dtype=dtype)
        input_ids = input_ids.to(device=device)
        attention_mask = attention_mask.to(device=device)
        image_grid_thw = image_grid_thw.to(device=device)
        generated_ids = self.generate(
            generation_config=self.generation_config,
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        output_text = self.tokenizer.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        del pixel_values
        del input_ids
        del attention_mask
        del image_grid_thw
        del generated_ids
        del generated_ids_trimmed
        gc.collect()
        torch.cuda.empty_cache()

        return (output_text, generated_ids) if return_generation_ids else output_text

    def warp_qwen2vl_lora(
        self, r, lora_alpha, freeze_vision_tower=True, lora_dropout=0.05
    ):
        target_modules = find_all_linear_modules(
            self.model, freeze_vision_tower=freeze_vision_tower
        )
        target_modules = patch_target_modules(
            self.model.config,
            freeze_vision_tower=freeze_vision_tower,
            target_modules=target_modules,
        )
        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.enable_input_require_grads()
