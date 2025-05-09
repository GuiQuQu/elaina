from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
import open_clip

from torch import distributed as dist

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

from utils.dist_variable import DistVarible
from utils.register import Register

HF_HUB_PREFIX = "hf-hub:"

def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        raise ValueError("Horovod is not supported in this version")
        # assert hvd is not None, 'Please install horovod'
        # if gather_with_grad:
        #     all_image_features = hvd.allgather(image_features)
        #     all_text_features = hvd.allgather(text_features)
        # else:
        #     with torch.no_grad():
        #         all_image_features = hvd.allgather(image_features)
        #         all_text_features = hvd.allgather(text_features)
        #     if not local_loss:
        #         # ensure grads for local rank when all_* features don't have a gradient
        #         gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
        #         gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
        #         gathered_image_features[rank] = image_features
        #         gathered_text_features[rank] = text_features
        #         all_image_features = torch.cat(gathered_image_features, dim=0)
        #         all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


def get_torch_dtype(model_dtype: str):
    if model_dtype == "bf16":
        return torch.bfloat16
    elif model_dtype == "fp16":
        return torch.float16
    elif model_dtype == "fp32":
        return torch.float32
    else:
        raise ValueError(f"model_dtype {model_dtype} is not supported")


@Register(name="eva02_clip")
class EVA02CLIP(nn.Module):
    def __init__(
        self,
        model_path: str,
        model_name: str = "EVA02-L-14",
        tokenizer_path: Optional[str] = None,
        file_name: str = "open_clip_pytorch_model.bin",
        precision: str = "fp32",
        device="cpu",
        local_loss = True,
        gather_with_grad = True,
    ):
        super().__init__()
        cpkt_file = model_path + "/" + file_name
        # 使用amp进行混合精度训练(fp32,fp16)的模型, 模型的初始化权重类型必须是fp32
        # 只有推理的模型模型才能转fp16进行推理
        self.model, self.preprocess_train, self.preprocess_val = (
            open_clip.create_model_and_transforms(
                model_name=model_name,
                pretrained=cpkt_file,
                device=device,
                precision=precision,
                output_dict=False,
            )
        )
        
        if tokenizer_path is None:
            tokenizer_path = model_path
        self.tokenizer = open_clip.get_tokenizer(HF_HUB_PREFIX + tokenizer_path)

        self.loss = ClipLoss(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=True,
            rank = DistVarible.rank,
            world_size=DistVarible.world_size,
        )

    def gradient_checkpointing_enable(self,gradient_checkpointing_kwargs = None):
        # 梯度检查无法使用，要想使用必须设置{"use_reentrant": false}
        # 但是这个模型在open_clip内部，然后内部用的视觉模型在timm里面
        # 自己不好去设置这个参数。
        self.model.set_grad_checkpointing(enable=True)

    def encode_image(self, image:torch.Tensor, normalize=True):
        return self.model.encode_image(image, normalize)

    def encode_text(self, text:torch.Tensor, normalize=True):
        return self.model.encode_text(text, normalize)

    def get_logits(self, image, text):
        """
            Args:
                image: [B,C,H,W]
                text: [B,L]
            Returns:
                logits_per_image: [B,B]
                logits_per_text: [B,B]
        """
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        image = image.to(device=device, dtype=dtype)
        text = text.to(device=device)
        return self.model.get_logits(image, text)

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
    ):
        image_features, text_features, logits_bias = self.model(image,text)
        loss = self.loss(image_features, text_features, logits_bias)

        return loss, (image_features, text_features, logits_bias)
    
    @torch.no_grad()
    def inference_forward(self,image, text):
        # [B,B]
        logits_per_image, _ = self.get_logits(image, text)
        # 取对角线元素
        
        score = torch.diag(logits_per_image)
        score = score.detach().to(torch.float32).cpu().numpy().tolist()
        # score = [{'score': s}for s in score]
        return None, score