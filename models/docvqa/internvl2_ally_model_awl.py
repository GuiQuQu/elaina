import torch
from torch import nn
from torch.nn import functional as F
from peft import get_peft_model, LoraConfig
from transformers.activations import ACT2FN

from models.docvqa.internvl2.configuration_internvl_chat import (
    InternVLChatConfig,
)
from models.docvqa.internvl2.modeling_internvl_chat import (
    InternVLChatModel,
)
from models.docvqa.internvl2.tokenization_internlm2 import (
    InternLM2Tokenizer,
)

from models.docvqa.internvl2.constant import IMG_CONTEXT_TOKEN
from models.automatic_weighted_loss import AutomaticWeightedLoss

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


class MLP(nn.Module):
    def __init__(self, name, sizes, act_fn="relu", add_bias: bool = False) -> None:
        super(MLP, self).__init__()
        self.name = name
        self.layers = nn.ModuleList([])
        for i in range(len(sizes) - 1):
            input_size = sizes[i]
            output_size = sizes[i + 1]
            self.layers.append(nn.Linear(input_size, output_size, bias=add_bias))
        self.act = ACT2FN[act_fn]

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.act(x)
        return x


class InternVL2AllyModel(nn.Module):
    def __init__(
        self,
        model_path,
        use_backbone_lora=0,
        use_llm_lora=0,
        freeze_vision_model=True,
        freeze_llm_model=False,
        frzzze_mlp=True,
        model_dtype="bf16",
    ) -> None:
        super().__init__()
        # load into cpu, in train will be transfer to gpu
        config = InternVLChatConfig.from_pretrained(model_path)
        self.model: InternVLChatModel = InternVLChatModel.from_pretrained(
            model_path,
            config=config,
            torch_dtype=get_torch_dtype(model_dtype),
            low_cpu_mem_usage=True,
            use_flash_attn=True,
        )
        self.tokenizer = InternLM2Tokenizer.from_pretrained(model_path)
        self.pad_id = self.tokenizer.pad_token_id
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(
            IMG_CONTEXT_TOKEN
        )

        hidden_dim = config.llm_config.hidden_size
        self.classify_mlp = MLP(
            name="classify_mlp",
            sizes=[hidden_dim, 512, 256, 1],
            act_fn="relu",
            add_bias=False,
        )
        self.classify_mlp = self.classify_mlp.to(get_torch_dtype(model_dtype))

        if freeze_vision_model:
            self.model.vision_model = self.model.vision_model.eval()
            _freeze_params(self.model.vision_model)
        if freeze_llm_model:
            self.model.language_model = self.model.language_model.eval()
            _freeze_params(self.model.language_model)
        if frzzze_mlp:
            _freeze_params(self.model.mlp1)

        # warpped model with lora
        if use_backbone_lora:
            self.wrap_backbone_lora(use_backbone_lora, 2 * use_backbone_lora)
        if use_llm_lora:
            self.wrap_llm_lora(
                use_llm_lora,
                2 * use_llm_lora,
            )

        self.awl = AutomaticWeightedLoss(2)
        self.count_parameters()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs
        )
        # vision model not support gradient checkpointing
        # self.model.vision_model.gradient_checkpointing_enable()

    def count_parameters(self):
        print(self)
        traing_parms = sum(p.numel() for p in self.parameters() if p.requires_grad)
        all_parms = sum(p.numel() for p in self.parameters())
        logger.info(
            f"training parameters: {traing_parms / 10**6} M, all parameters: {all_parms / 10**6} M"
        )

    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.model.vision_model = get_peft_model(self.model.vision_model, lora_config)
        # self.model.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        # Determine the target modules based on the architecture of the language model
        #'InternLM2ForCausalLM':
        target_modules = [
            "attention.wqkv",
            "attention.wo",
            "feed_forward.w1",
            "feed_forward.w2",
            "feed_forward.w3",
        ]
        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type="CAUSAL_LM",
        )
        self.model.language_model = get_peft_model(
            self.model.language_model, lora_config
        )
        self.model.language_model.enable_input_require_grads()
        # self.model.language_model.print_trainable_parameters()

    def forward(
        self,
        pixel_values,
        vqa_input_ids,
        vqa_attention_mask,
        image_flags,
        classify_input_ids,
        classify_attention_mask,
        vqa_label=None,
        cls_label=None,
    ):

        loss = None
        pixel_values = pixel_values.to(device=self.model.device, dtype=self.model.dtype)
        vqa_input_ids = vqa_input_ids.to(device=self.model.device)
        vqa_attention_mask = vqa_attention_mask.to(device=self.model.device)
        image_flags = image_flags.to(device=self.model.device)
        classify_input_ids = classify_input_ids.to(device=self.model.device)
        classify_attention_mask = classify_attention_mask.to(device=self.model.device)

        cls_loss, cls_outputs = self.classify_forward(
            pixel_values,
            classify_input_ids,
            classify_attention_mask,
            image_flags,
            cls_label,
        )
        vqa_loss, vqa_outputs = self.vqa_forward(
            pixel_values,
            vqa_input_ids,
            vqa_attention_mask,
            image_flags,
            vqa_label,
        )
        # 两个任务的loss权重
        if cls_loss and vqa_loss:
            loss = self.awl(cls_loss, vqa_loss)

        return loss, cls_outputs

    def classify_forward(
        self,
        pixel_values,
        classify_input_ids,
        classify_attention_mask,
        image_flags,
        cls_label=None,
    ):
        cls_loss = None
        transformer_output = self.model(
            pixel_values=pixel_values,
            input_ids=classify_input_ids,
            attention_mask=classify_attention_mask,
            image_flags=image_flags,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = transformer_output.hidden_states[-1]
        llm_embedding = self.get_last_one_hidden_states(
            hidden_states, classify_input_ids, padding_on_left=True
        )
        logits = self.classify_mlp(llm_embedding)  # [B,1]
        outputs = F.sigmoid(logits)
        outputs = outputs.view(-1).detach().to(torch.float32).cpu().numpy().tolist()
        B, _ = logits.size()
        if cls_label is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            cls_label = cls_label.view(B, 1).to(
                device=logits.device, dtype=logits.dtype
            )
            cls_loss = loss_fct(logits, cls_label)
        return cls_loss, outputs

    def vqa_forward(
        self,
        pixel_values,
        vqa_input_ids,
        vqa_attention_mask,
        image_flags,
        vqa_label=None,
    ):

        transformer_output = self.model(
            pixel_values=pixel_values,
            input_ids=vqa_input_ids,
            attention_mask=vqa_attention_mask,
            image_flags=image_flags,
            labels=vqa_label,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        vqa_loss = transformer_output.loss
        return vqa_loss, transformer_output

    def get_last_one_hidden_states(
        self, hidden_states, input_ids, padding_on_left=False
    ):
        bsz, _, _ = hidden_states.size()

        padding_on_left = padding_on_left

        if padding_on_left:
            return hidden_states[:, -1, :].view(bsz, -1)
        else:
            device = input_ids.device
            temp_pad = torch.tensor([self.pad_id] * bsz).long().view(bsz, -1)  # [bsz,1]
            temp_pad = temp_pad.to(device)
            temp_input_ids = torch.cat([input_ids, temp_pad], dim=1)
            bsz_idx, last_idx = [], []
            for i in range(bsz):
                temp = temp_input_ids[i]
                bsz_idx.append(i)
                t = torch.nonzero(temp == self.pad_id, as_tuple=True)
                last_idx.append(t[0].min() - 1)

            assert all([i >= 0 for i in last_idx])

            bsz_idx = torch.tensor(bsz_idx).to(device)
            last_idx = torch.tensor(last_idx).to(device)
            return hidden_states[bsz_idx, last_idx, :]

    def batch_chat(self, *arg, **kwargs):
        return self.model.batch_chat(*arg, **kwargs)

    def chat(self, *arg, **kwargs):
        return self.model.chat(*arg, **kwargs)

    def generate(self, *arg, **kwargs):
        return self.model.generate(*arg, **kwargs)
