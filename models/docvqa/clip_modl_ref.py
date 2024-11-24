from typing import List, Optional
import logging
from PIL import Image
import torch
from torch import nn
from timm.loss import LabelSmoothingCrossEntropy
import torch.functional as F

import open_clip

HF_HUB_PREFIX = "hf-hub:"


def get_cast_type(model) -> torch.dtype:
    if isinstance(model, torch.nn.Module):
        return next(model.parameters()).dtype
    else:
        return None


def get_cast_device(model) -> torch.device:
    if isinstance(model, torch.nn.Module):
        return next(model.parameters()).device
    else:
        return None


def load_model_tokenizer_transform(
    model_name: str,
    cpkt_path: str,
    tokenizer_path: str = None,
    file_name: str = "open_clip_pytorch_model.bin",
    precision: str = "fp32",
    device="cpu",
):
    cpkt_file = cpkt_path + "/" + file_name
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name=model_name, pretrained=cpkt_file, device=device, precision=precision
    )
    if tokenizer_path is None:
        tokenizer_path = cpkt_path
    tokenizer = open_clip.get_tokenizer(HF_HUB_PREFIX + tokenizer_path)

    return model, preprocess_train, preprocess_val, tokenizer


# def gather_features(
#         image_features,
#         text_features,
#         local_loss=False,
#         gather_with_grad=False,
#         rank=0,
#         world_size=1,
#         use_horovod=False
# ):
#     assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
#     if use_horovod:
#         assert hvd is not None, 'Please install horovod'
#         if gather_with_grad:
#             all_image_features = hvd.allgather(image_features)
#             all_text_features = hvd.allgather(text_features)
#         else:
#             with torch.no_grad():
#                 all_image_features = hvd.allgather(image_features)
#                 all_text_features = hvd.allgather(text_features)
#             if not local_loss:
#                 # ensure grads for local rank when all_* features don't have a gradient
#                 gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
#                 gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
#                 gathered_image_features[rank] = image_features
#                 gathered_text_features[rank] = text_features
#                 all_image_features = torch.cat(gathered_image_features, dim=0)
#                 all_text_features = torch.cat(gathered_text_features, dim=0)
#     else:
#         # We gather tensors from all gpus
#         if gather_with_grad:
#             all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
#             all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
#             # all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features, async_op=True), dim=0)
#             # all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features, async_op=True), dim=0)
#         else:
#             gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
#             gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
#             dist.all_gather(gathered_image_features, image_features)
#             dist.all_gather(gathered_text_features, text_features)
#             if not local_loss:
#                 # ensure grads for local rank when all_* features don't have a gradient
#                 gathered_image_features[rank] = image_features
#                 gathered_text_features[rank] = text_features
#             all_image_features = torch.cat(gathered_image_features, dim=0)
#             all_text_features = torch.cat(gathered_text_features, dim=0)

#     return all_image_features, all_text_features


class ClipLoss(nn.Module):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
        smoothing=0.0,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.label_smoothing_cross_entropy = (
            LabelSmoothingCrossEntropy(smoothing=smoothing) if smoothing > 0 else None
        )
        self.loss_fct = nn.CrossEntropyLoss()
        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale=1.0):
        device = image_features.device
        if self.world_size > 1:
            raise ValueError("not support world size > 1")
        # all_image_features, all_text_features = gather_features(
        #     image_features,
        #     text_features,
        #     self.local_loss,
        #     self.gather_with_grad,
        #     self.rank,
        #     self.world_size,
        #     self.use_horovod,
        # )

        # if self.local_loss:
        #     logits_per_image = logit_scale * image_features @ all_text_features.T
        #     logits_per_text = logit_scale * text_features @ all_image_features.T
        # else:
        #     logits_per_image = (
        #         logit_scale * all_image_features @ all_text_features.T
        #     )
        #     logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        if self.label_smoothing_cross_entropy:
            total_loss = (
                self.label_smoothing_cross_entropy(logits_per_image, labels)
                + self.label_smoothing_cross_entropy(logits_per_text, labels)
            ) / 2
        else:
            total_loss = (
                self.loss_fct(logits_per_image, labels)
                + self.loss_fct(logits_per_text, labels)
            ) / 2

        acc = None
        i2t_acc = (logits_per_image.argmax(-1) == labels).sum() / len(logits_per_image)
        t2i_acc = (logits_per_text.argmax(-1) == labels).sum() / len(logits_per_text)
        acc = {"i2t": i2t_acc, "t2i": t2i_acc}
        return total_loss, acc


class TripletLoss(nn.Module):
    def __init__(self, margin: float):
        super().__init__()
        self.margin = margin

    def forward(self, anchor_features, positive_features, negitive_features):
        dist_a_p = torch.sum(anchor_features * positive_features, dim=-1)  # [bsz]
        dist_a_n = torch.sum(anchor_features * negitive_features, dim=-1)  # [bsz]
        loss = dist_a_p - dist_a_n + self.margin
        loss = torch.where(loss > 0, loss, 0)
        return loss.mean()


def test_CLIP_model():
    logging.basicConfig(level=logging.INFO)
    model_name = "EVA02-L-14"
    cpkt_path = "../pretrain-model/eva02_large_patch14_clip_224.merged2b_s4b_b131k"
    cpkt_file = cpkt_path + "/open_clip_pytorch_model.bin"
    precision = "fp32"
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name=model_name, pretrained=cpkt_file, device="cpu", precision=precision
    )
    print(type(model))
    print(next(model.parameters()))
    tokenizer = open_clip.get_tokenizer(HF_HUB_PREFIX + cpkt_path)
    print(tokenizer)

    image = preprocess_val(Image.open("../CLIP.png")).unsqueeze(0)
    text = tokenizer(["a diagram", "a dog", "a cat"])
    input_device = get_cast_device(model)
    image = image.to(input_device)
    text = text.to(input_device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        print("Label probs:", text_probs)


class EVA02CLIP(nn.Module):
    def __init__(
        self,
        cpkt_path: str,
        model_name: str = "EVA02-L-14",
        tokenizer_path: Optional[str] = None,
        file_name: str = "open_clip_pytorch_model.bin",
        precision: str = "fp32",
        device="cpu",
    ):
        super().__init__()
        cpkt_file = cpkt_path + "/" + file_name
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
            tokenizer_path = cpkt_path
        self.tokenizer = open_clip.get_tokenizer(HF_HUB_PREFIX + tokenizer_path)

        self.loss = ClipLoss(
            local_loss=True,
            gather_with_grad=False,
            cache_labels=True,
            rank = 0,
            world_size=0,
        )

    def forward(self, image, text):
        pass

    def encode_image(self, image, normalize=True):
        return self.model.encode_image(image, normalize)

    def encode_text(self, text, normalize=True):
        return self.model.encode_text(text, normalize)

    def get_logits(self, image, text):
        return self.model.get_logits(image, text)

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
    ):
        image_features, text_features, logits_bias = self.model(image,text)



def test_TripletLoss():
    loss_fct = TripletLoss(margin=0.5)
    af = torch.rand(5, 10)
    pf = torch.rand(5, 10)
    nf = torch.rand(5, 10)
    loss = loss_fct(af, pf, nf)
    print(loss)


if __name__ == "__main__":
    test_TripletLoss()
