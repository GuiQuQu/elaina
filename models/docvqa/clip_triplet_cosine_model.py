from typing import List, Optional
import torch
from torch import nn
from torch.nn import functional as F
import open_clip

from torch.nn import MarginRankingLoss

from utils.dist_variable import DistVarible
from utils.register import Register

HF_HUB_PREFIX = "hf-hub:"


def get_torch_dtype(model_dtype: str):
    if model_dtype == "bf16":
        return torch.bfloat16
    elif model_dtype == "fp16":
        return torch.float16
    elif model_dtype == "fp32":
        return torch.float32
    else:
        raise ValueError(f"model_dtype {model_dtype} is not supported")


def triplet_loss(triplet, margin=0.2):
    loss = 0
    dist_a_p = triplet[:, 0]
    dist_a_n = triplet[:, 1]
    loss = dist_a_p - dist_a_n + margin
    loss[loss < 0] = 0
    cnt = (loss > 0).sum()
    return loss.sum() / cnt


def euclidean_distance(x, y, do_sqrt=True):
    """
    Args:
        x: [m, D]
        y: [n, D]
    Returns:
        distance: [m, n]
    """
    sq_x = x**2
    sq_y = y**2
    sum_sq_x = torch.sum(sq_x, dim=1).unsqueeze(1)  # m -> [m, 1]
    sum_sq_y = torch.sum(sq_y, dim=1).unsqueeze(0)  # n -> [1, n]
    yt = y.transpose(0, 1)
    ret = sum_sq_x - 2 * torch.matmul(x, yt) + sum_sq_y
    if do_sqrt:
        ret = torch.sqrt(ret)
    return ret

def cosine_distance(x,y):
    """
    Args:
        x: [m, D]
        y: [n, D]
    Returns:
        distance: [m, n]
    """
    return 1 - cosine_similarity(x, y)

def cosine_similarity(x, y):
    """
    Args:
        x: [m, D]
        y: [n, D]
    Returns:
        similarity: [m, n]
    """
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    return torch.matmul(x, y.transpose(0, 1))

def cosine_similarity_same_shape(x,y):
    """
    给定两个相同shape的向量，计算每一行的两个向量两两余弦相似度
    Args:
        x: [m, D]
        y: [m, D]
    Returns:
        similarity: [m, 1]
    """
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    return torch.sum(x * y, dim=-1).view(-1, 1)

def cosine_distance_same_shape(x,y):
    """
    给定两个相同shape的向量，计算每一行的两个向量两两余弦距离
    Args:
        x: [m, D]
        y: [m, D]
    Returns:
        distance: [m, 1]
    """
    return 1 - cosine_similarity_same_shape(x, y)


@Register(name="eva02_clip_for_triplet_consine")
class EVA02CLIPForTriplet(nn.Module):
    def __init__(
        self,
        model_path: str,
        model_name: str = "EVA02-L-14",
        tokenizer_path: Optional[str] = None,
        file_name: str = "open_clip_pytorch_model.bin",
        precision: str = "fp32",
        device="cpu",
        margin=0.2,
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
        self.margin = margin
        self.loss_fct = MarginRankingLoss(margin=margin)
        # self.max_negative_images = max_negative_images

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        # 梯度检查无法使用，要想使用必须设置{"use_reentrant": false}
        # 但是这个模型在open_clip内部，然后内部用的视觉模型在timm里面
        # 自己不好去设置这个参数。
        self.model.set_grad_checkpointing(enable=True)

    def encode_image(self, image: torch.Tensor, normalize=True):
        return self.model.encode_image(image, normalize)

    def encode_text(self, text: torch.Tensor, normalize=True):
        return self.model.encode_text(text, normalize)

    def forward(
        self,
        anchor_input_ids: torch.Tensor,
        positive_image: torch.Tensor,
        negative_images: List[torch.Tensor],
    ):
        """
        余弦距离的triplet loss
        Args:
            anchor_input_ids: [B,L]
            positive_image: [B,C,H,W]
            negative_images: the List length is N ,elem is torch.Tensor [K,C,H,W]
            and every K maybe different,
            K1 + K2 + ... KB = N
        Returns:
            loss: scalar tensor
            triplet: [B*N, 2]
        """
        anchor_embed = self.encode_text(anchor_input_ids)  # [B, D]
        positive_embed = self.encode_image(positive_image)  # [B, D]
        negative_images = torch.cat(negative_images, dim=0)  # [N, C, H, W]
        negative_embeds = self.encode_image(negative_images)  # [N, D]
        B, D = anchor_embed.size()
        N = negative_embeds.size(0)
        # 计算所有的d(a,p)
        dist_ap = cosine_distance_same_shape(anchor_embed, positive_embed)  # [B,1]
        dist_ap = dist_ap.repeat(1, N)  # [B, N]
        dist_an = cosine_distance(anchor_embed, negative_embeds)  # [B, N]
        # 共有B*N个三元组
        dist_ap = dist_ap.view(-1)
        dist_an = dist_an.view(-1)   
        loss = self.loss_fct(dist_ap, dist_an, -1 * torch.ones_like(dist_ap))
        return loss, None

    @torch.no_grad()
    def inference_forward(self, image, text):
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        image = image.to(device=device, dtype=dtype)
        text = text.to(device=device)
        image_embed = self.encode_image(image, normalize=True)
        text_embed = self.encode_text(text, normalize=True)
        cos_sim = cosine_similarity_same_shape(image_embed, text_embed).view(-1)
        score = cos_sim.detach().to(torch.float32).cpu().numpy().tolist()
        return None, score
