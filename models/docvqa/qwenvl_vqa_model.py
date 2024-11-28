from typing import List
import torch
from torch import nn
from torch.nn import functional as F
from peft import get_peft_model, LoraConfig
from transformers import PreTrainedModel

from models.docvqa.qwenvl.configuration_qwen import QWenConfig
from models.docvqa.qwenvl.modeling_qwen import QWenLMHeadModel
from models.docvqa.qwenvl.tokenization_qwen import QWenTokenizer


from logger import logger
from utils.register import Register


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


# lora 的优先级比freeze的优先级高
# 就算freeze了，但是lora的设置依然是有效的
@Register(name="qwenvl_vqa_model")
class QwenVLVQAModel(nn.Module):
    def __init__(
        self,
        model_path,
        gradient_checkpointing: bool = True,
        use_qwenvl_lora: int = 8,
        freeze_vision_model=True,
        model_dtype="bf16",
    ) -> None:
        super().__init__()
        # load into cpu, in train will be transfer to gpu
        config = QWenConfig.from_pretrained(model_path)
        self.model: QWenLMHeadModel = QWenLMHeadModel.from_pretrained(
            model_path,
            config=config,
            torch_dtype=get_torch_dtype(model_dtype),
            low_cpu_mem_usage=True,
        )
        self.generation_config = self.model.generation_config
        self.tokenizer:QWenTokenizer = QWenTokenizer.from_pretrained(model_path)
        self.gradient_checkpointing = gradient_checkpointing
        
        # qwenvl 模型在这里会设置梯度检查，如果hf_trainer也会设置梯度检查，
        # 因此这个函数实际会被调用两边，请注意
        if gradient_checkpointing:
            self.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.pad_id = self.tokenizer.eod_id

        if freeze_vision_model:
            self.model.transformer.visual = self.model.transformer.visual.eval()
            _freeze_params(self.model.transformer.visual)

        # warpped model with lora
        if use_qwenvl_lora:
            self.wrap_qwenvl_lora(use_qwenvl_lora, 2 * use_qwenvl_lora)

        self.count_parameters()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.transformer.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs
        )

    def count_parameters(self):
        print(self)
        traing_parms = sum(p.numel() for p in self.parameters() if p.requires_grad)
        all_parms = sum(p.numel() for p in self.parameters())
        logger.info(
            f"training parameters: {traing_parms / 10**6} M, all parameters: {all_parms / 10**6} M"
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        # test_input_ids,
        # test_attention_mask,
        labels,
    ):
        """
        Args:
            input_ids: torch.Tensor, [B, L]
            attention_mask: torch.Tensor, [B, L]
            input_ids 中包含了image_path编码后的内容，可以通过decode进行还原
            qwenvl内部的visual encoder会自行处理
        """
        loss = None
        input_ids = input_ids.to(device=self.model.device)
        attention_mask = attention_mask.to(device=self.model.device)

        transformer_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=not self.training,
            output_hidden_states=True,
            return_dict=True,
        )
        loss = transformer_output.loss
        return loss, transformer_output

    def wrap_qwenvl_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        # Determine the target modules based on the architecture of the language model
        #'InternLM2ForCausalLM':
        target_modules = ["c_attn", "attn.c_proj", "w1", "w2"]
        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        if self.gradient_checkpointing:
            self.model.enable_input_require_grads()

    def chat(self, *arg, **kwargs):
        return self.model.chat(*arg, **kwargs)

    def generate(self, *arg, **kwargs):
        return self.model.generate(*arg, **kwargs)

    def inference_forward(
        self,
        test_input_ids,
        test_attention_mask,
    ):
        test_input_ids = test_input_ids.to(device=self.model.device)
        test_attention_mask = test_attention_mask.to(device=self.model.device)

        stop_words_ids = get_stop_words_ids(
            self.generation_config.chat_format, self.tokenizer
        )
        generated_ids = self.generate(
            input_ids=test_input_ids,
            attention_mask=test_attention_mask,
            stop_words_ids=stop_words_ids,
            return_dict_in_generate=False,
            generation_config=self.generation_config,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(test_input_ids, generated_ids)
        ]
        output_text = self.tokenizer.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        
        return None, output_text


def get_stop_words_ids(chat_format, tokenizer):
    if chat_format == "raw":
        stop_words_ids = [tokenizer.encode("Human:"), [tokenizer.eod_id]]
    elif chat_format == "chatml":
        stop_words_ids = [[tokenizer.im_end_id], [tokenizer.im_start_id]]
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")
    return stop_words_ids
