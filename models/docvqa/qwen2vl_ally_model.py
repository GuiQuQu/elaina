from typing import List
import torch
from torch import nn
import torch.nn.functional as F
from transformers import (
    # Qwen2VLForConditionalGeneration,
    Qwen2VLConfig,
    Qwen2Tokenizer,
    GenerationConfig,
)
from models.docvqa.qwen2vl import Qwen2VLForConditionalGenerationForAlly
from transformers.activations import ACT2FN

from utils.register import Register
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


@Register(name="qwen2vl_ally_model")
class Qwen2VLAllyModel(nn.Module):
    def __init__(
        self,
        model_path,
        freeze_vision_model=True,
        freeze_llm_model=False,
        model_dtype="bf16",
    ) -> None:
        super().__init__()
        config = Qwen2VLConfig.from_pretrained(model_path)
        if self.training:
            config.use_cache = False
        self.generation_config = GenerationConfig.from_pretrained(model_path)
        self.model: Qwen2VLForConditionalGenerationForAlly = (
            Qwen2VLForConditionalGenerationForAlly.from_pretrained(
                model_path,
                config=config,
                torch_dtype=get_torch_dtype(model_dtype),
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2",
            )
        )

        self.tokenizer: Qwen2Tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        # self.tokenizer.padding_side = "left"
        self.pad_id = self.tokenizer.pad_token_id

        hidden_dim = config.hidden_size
        self.classify_mlp = MLP(
            name="classify_mlp",
            sizes=[hidden_dim, 512, 256, 1],
            act_fn="relu",
            add_bias=False,
        )
        self.classify_mlp = self.classify_mlp.to(get_torch_dtype(model_dtype))
        self.awl = AutomaticWeightedLoss(2)

        if freeze_vision_model:
            self.model.visual = self.model.visual.eval()
            _freeze_params(self.model.visual)
        if freeze_llm_model:
            self.model.model = self.model.model.eval()
            _freeze_params(self.model.model)

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
        classify_pixel_values,
        classify_image_grid_thw,
        classify_input_ids,
        classify_attention_mask,
        vqa_pixel_values,
        vqa_image_grid_thw,
        vqa_input_ids,
        vqa_attention_mask,
        classify_label=None,
        vqa_label=None,
    ):

        loss = None
        vqa_pixel_values = vqa_pixel_values.to(device=self.model.device, dtype=self.model.dtype)
        vqa_image_grid_thw = vqa_image_grid_thw.to(device=self.model.device)
        vqa_input_ids = vqa_input_ids.to(device=self.model.device)
        vqa_attention_mask = vqa_attention_mask.to(device=self.model.device)
        
        classify_pixel_values = classify_pixel_values.to(device=self.model.device, dtype=self.model.dtype)
        classify_image_grid_thw = classify_image_grid_thw.to(device=self.model.device)
        classify_input_ids = classify_input_ids.to(device=self.model.device)
        classify_attention_mask = classify_attention_mask.to(device=self.model.device)

        cls_loss, cls_outputs = self.classify_forward(  # cls_loss size: [batch_size,1]
            classify_pixel_values,
            classify_input_ids,
            classify_attention_mask,
            classify_image_grid_thw,
            classify_label,
        )
        vqa_loss, vqa_outputs = self.vqa_forward(  # vqa_loss size: [batch_size*seq_len-1],因为存在shift
            vqa_pixel_values,
            vqa_input_ids,
            vqa_attention_mask,
            vqa_image_grid_thw,
            vqa_label,
        )
        if vqa_loss is not None and cls_loss is not None:
            B, L = vqa_input_ids.size()
            dtype = cls_loss.dtype
            classify_label = classify_label.view(B).to(dtype=dtype)
            cls_loss = cls_loss.view(B)
            vqa_loss = vqa_loss.view(B, L-1)
            vqa_cnt = torch.where(vqa_loss == 0, 0, 1).sum(dim=1) # [B]
            vqa_loss = vqa_loss.sum(dim=1) / vqa_cnt
            
            # 采用自动权重
            loss = torch.zeros(B, dtype=dtype).to(device=cls_loss.device)
            for i in range(B):
                if classify_label[i] == 0:
                    loss[i] = cls_loss[i]
                else:
                    loss[i] = self.awl(cls_loss[i], vqa_loss[i])
            loss = loss.mean()            

        return loss, cls_outputs

    def vqa_forward(
        self,
        vqa_pixel_values,
        vqa_input_ids,
        vqa_attention_mask,
        vqa_image_grid_thw,
        vqa_label=None,
    ):

        device = self.model.device
        dtype = self.model.dtype
        loss = None
        vqa_pixel_values = vqa_pixel_values.to(device=device, dtype=dtype)
        vqa_input_ids = vqa_input_ids.to(device=device)
        vqa_attention_mask = vqa_attention_mask.to(device=device)
        vqa_image_grid_thw = vqa_image_grid_thw.to(device=device)
        outputs = self.model(
            pixel_values=vqa_pixel_values,
            input_ids=vqa_input_ids,
            attention_mask=vqa_attention_mask,
            image_grid_thw=vqa_image_grid_thw,
            labels=vqa_label,
            output_hidden_states=True,
            return_dict=True,
        )
        loss = outputs.loss
        return loss, outputs

    def classify_forward(
        self,
        classify_pixel_values,
        classify_input_ids,
        classify_attention_mask,
        classify_image_grid_thw,
        classify_label=None,
    ):
        device = self.model.device
        dtype = self.model.dtype
        cls_loss = None
        classify_pixel_values = classify_pixel_values.to(device=device, dtype=dtype)
        classify_input_ids = classify_input_ids.to(device=device)
        classify_attention_mask = classify_attention_mask.to(device=device)
        classify_image_grid_thw = classify_image_grid_thw.to(device=device)
        qwen2vl_outputs = self.model(
            pixel_values=classify_pixel_values,
            input_ids=classify_input_ids,
            attention_mask=classify_attention_mask,
            image_grid_thw=classify_image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
        )
        # loss = outputs.loss
        hidden_states = qwen2vl_outputs.hidden_states[-1]
        # qwen2 的 tokenizer 默认都是左padding
        llm_embedding = self.get_last_one_hidden_states(
            hidden_states, classify_attention_mask, padding_on_left = True,
        )
        logits = self.classify_mlp(llm_embedding)
        outputs = F.sigmoid(logits)
        outputs = outputs.view(-1).detach().to(torch.float32).cpu().numpy().tolist()
        B, _ = logits.size()
        if classify_label is not None:
            loss_fct = nn.BCEWithLogitsLoss(reduction="none")
            classify_label = classify_label.view(B, 1).to(
                device=logits.device, dtype=logits.dtype
            )
            cls_loss = loss_fct(logits, classify_label)
        return cls_loss, outputs

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

    def generate(self, *arg, **kwargs):
        return self.model.generate(*arg, **kwargs)

    def inference_forward(
        self,
        test_vqa_pixel_values,
        test_vqa_input_ids,
        test_vqa_attention_mask,
        test_classify_pixel_values,
        test_classify_input_ids,
        test_classify_attention_mask,
        test_vqa_image_grid_thw=None,
        # test_vqa_video_grid_thw=None,
        test_classify_image_grid_thw=None,
        # test_classify_video_grid_thw=None,
    ):
        
        vqa_outputs = self.batch_chat(
            pixel_values=test_vqa_pixel_values,
            input_ids=test_vqa_input_ids,
            attention_mask=test_vqa_attention_mask,
            image_grid_thw=test_vqa_image_grid_thw,
        )
        _ , classify_ouputs = self.classify_forward(
            classify_pixel_values=test_classify_pixel_values,
            classify_input_ids=test_classify_input_ids,
            classify_attention_mask=test_classify_attention_mask,
            classify_image_grid_thw=test_classify_image_grid_thw,
        )
        result = []
        for cls_score, vqa_text in zip(classify_ouputs, vqa_outputs):
            result.append({'score': cls_score, 'resp': vqa_text})

        return result

    def batch_chat(
        self,
        pixel_values,
        input_ids,
        attention_mask,
        image_grid_thw=None,
        video_grid_thw=None,
        max_new_tokens=128,
    ) -> List[str]:
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
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        output_text = self.tokenizer.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text
