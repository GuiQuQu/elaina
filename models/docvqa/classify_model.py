# internvl2(2b version) 做embedding model 来进行分类，lora内包裹

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


def _freeze_params(module):
    for param in module.parameters():
        param.requires_grad = False

# lora 的优先级比freeze的优先级高
# 就算freeze了，但是lora的设置依然是有效的
class InternVL2ClassifyModel(nn.Module):
    def __init__(
        self,
        model_path,
        use_backbone_lora=0,
        use_llm_lora=0,
        freeze_vision_model=True,
        freeze_llm_model=False,
        frzzze_mlp=True,
        model_dtype="bf16",
        load_from_ckpt=None,
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
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        hidden_dim = config.llm_config.hidden_size
        # mlp
        self.classify_mlp = MLP(
            name="clsasify_mlp",
            sizes=[hidden_dim, 512, 256, 1],
            act_fn="relu",
            add_bias=False,
        )

        if load_from_ckpt is not None:
            # load ckpt from local
            pass

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
            self.wrap_backbone_lora(use_backbone_lora, 2*use_backbone_lora)
        if use_llm_lora:
            self.wrap_llm_lora(use_llm_lora, 2*use_llm_lora, )

        self.count_parameters()

    def gradient_checkpointing_enable(self,gradient_checkpointing_kwargs=None):
        self.model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        # vision model not support gradient checkpointing
        # self.model.vision_model.gradient_checkpointing_enable()

    def wrapped_by_peft(self, lora_config):
        lora_config = LoraConfig(**lora_config)
        lora_model = get_peft_model(self.model, lora_config)
        return lora_model

    def count_parameters(self):
        print(self)
        traing_parms = sum(p.numel() for p in self.parameters() if p.requires_grad)
        all_parms = sum(p.numel() for p in self.parameters())
        logger.info(
            f"training parameters: {traing_parms / 10**6} M, all parameters: {all_parms / 10**6} M"
        )

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

    def forward(
        self,
        # pixel_values,
        test_pixel_values,
        # input_ids,
        # attention_mask,
        image_flags,
        test_input_ids,
        test_attention_mask,
        cls_label,
    ):
        """
        Args:
            pixel_values: torch.Tensor, [N, 3, 448, 448]
            input_ids: torch.Tensor, [B, L]
            attention_mask: torch.Tensor, [B, L]
            image_flags: torch.Tensor, [N]
            cls_label: torch.Tensor, [B]
        """
        loss = None
        outputs = None
        # breakpoint()
        test_pixel_values = test_pixel_values.to(device=self.model.device)
        transformer_output = self.model(
            pixel_values=test_pixel_values,
            input_ids=test_input_ids,
            attention_mask=test_attention_mask,
            image_flags=image_flags,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = transformer_output.hidden_states[-1]
        llm_embedding = self.get_last_one_hidden_states(
            hidden_states, test_input_ids, padding_on_left=True
        )
        # mlp
        logits = self.classify_mlp(llm_embedding)  # [B,1]
        outputs = F.sigmoid(logits)  # [B,1]
        outputs = outputs.view(-1).detach().to(torch.float32).cpu().numpy().tolist()
        B, _ = logits.size()
        if cls_label is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            cls_label = cls_label.view(B, 1).to(device=logits.device,dtype=logits.dtype)
            loss = loss_fct(logits, cls_label)
        return loss, outputs

    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.model.vision_model = get_peft_model(self.model.vision_model, lora_config)
        # self.model.vision_model.print_trainable_parameters()
    
    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        # Determine the target modules based on the architecture of the language model
        #'InternLM2ForCausalLM':
        target_modules = ['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.model.language_model = get_peft_model(self.model.language_model, lora_config)
        self.model.language_model.enable_input_require_grads()
        # self.model.language_model.print_trainable_parameters()
