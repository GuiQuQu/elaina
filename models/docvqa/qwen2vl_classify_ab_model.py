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


@Register(name="qwen2vl_classify_ab_model")
class Qwen2VLClassifyABModel(nn.Module):
    def __init__(
        self,
        model_path,
        a_token="A",
        b_token="B",
        token_position=-1,
        warp_qwen2vl_lora: int = 0,
        freeze_vision_model=True,
        freeze_llm_model=False,
        model_dtype="bf16",
    ) -> None:
        super().__init__()
        config = Qwen2VLConfig.from_pretrained(model_path)
        config.use_cache = False
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
        self.a_token_id = self.tokenizer.convert_tokens_to_ids(a_token)
        self.b_token_id = self.tokenizer.convert_tokens_to_ids(b_token)
        self.token_position = token_position

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
        train_pixel_values,
        train_input_ids,
        train_attention_mask,
        train_image_grid_thw,
        train_labels=None,
    ):
        
        device = self.model.device
        dtype = self.model.dtype
        loss = None
        train_pixel_values = train_pixel_values.to(device=device, dtype=dtype)
        train_input_ids = train_input_ids.to(device=device)
        train_attention_mask = train_attention_mask.to(device=device)
        train_image_grid_thw = train_image_grid_thw.to(device=device)
        if isinstance(train_labels, torch.Tensor):
            train_labels = train_labels.to(device=device)
        outputs = self.model(
            pixel_values=train_pixel_values,
            input_ids=train_input_ids,
            attention_mask=train_attention_mask,
            image_grid_thw=train_image_grid_thw,
            labels=train_labels,
            output_hidden_states=True,
            return_dict=True,
        )
        loss = outputs.loss
        logits = outputs.logits
        bsz = logits.size(0)
        padding_side = "left"
        last_idx = self.get_last_input_idx(
            train_input_ids, padding_side=padding_side, wanted_last_idx=self.token_position
        )
        bsz_idx = torch.arange(bsz)
        score = logits[bsz_idx, last_idx, :]  # [bsz, vocal_size]
        score = score[:, [self.b_token_id, self.a_token_id]]
        score = nn.Softmax(dim=-1)(score)  # [bsz, 2]
        score = score[:, 1]  # [bsz]
        score = score.detach().to(torch.float32).cpu().numpy().tolist()
        return loss, score

    def get_last_input_idx(
        self, input_ids, padding_side="right", wanted_last_idx: int = -1
    ):
        pass
        bsz, _ = input_ids.shape
        if padding_side == "left":
            return torch.tensor([wanted_last_idx] * bsz)
        elif padding_side == "right":
            device = input_ids.device
            temp_pad = torch.tensor([self.pad_id] * bsz).long().view(bsz, -1)  # [bsz,1]
            temp_pad = temp_pad.to(device)
            temp_input_ids = torch.cat([input_ids, temp_pad], dim=1)
            last_idx = []
            for i in range(bsz):
                temp = temp_input_ids[i]
                #
                t = torch.nonzero(temp == self.pad_id, as_tuple=True)
                last_idx.append(t[0].min() - wanted_last_idx)
            assert all([i >= 0 for i in last_idx])
            return torch.stack(last_idx)
        else:
            raise ValueError(f"padding_side {padding_side} is not supported")

    def inference_forward(
        self,
        pixel_values,
        input_ids,
        attention_mask,
        image_grid_thw=None,
    ):
        return self.forward(
            train_pixel_values=pixel_values,
            train_input_ids=input_ids,
            train_attention_mask=attention_mask,
            train_image_grid_thw=image_grid_thw,
        )

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
