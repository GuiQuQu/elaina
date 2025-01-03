import sys
from typing import List

import torch
from transformers.trainer_pt_utils import LabelSmoother
from transformers import Qwen2Tokenizer, Qwen2VLImageProcessor

from dataset.qwen2vl.template import Qwen2VLTemplate

IGNORE_INDEX_ID = LabelSmoother.ignore_index


def replace_text_func(
    texts: List[str],
    image_processor: Qwen2VLImageProcessor,
    image_grid_thw=None,
    video_grid_thw=None,
) -> List[str]:
    if image_grid_thw is not None:
        merge_length = image_processor.merge_size**2
        index = 0
        for i in range(len(texts)):
            while "<|image_pad|>" in texts[i]:
                texts[i] = texts[i].replace(
                    "<|image_pad|>",
                    "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length),
                    1,
                )
                index += 1
            texts[i] = texts[i].replace("<|placeholder|>", "<|image_pad|>")

    if video_grid_thw is not None:
        merge_length = image_processor.merge_size**2
        index = 0
        for i in range(len(texts)):
            while "<|video_pad|>" in texts[i]:
                texts[i] = texts[i].replace(
                    "<|video_pad|>",
                    "<|placeholder|>" * (video_grid_thw[index].prod() // merge_length),
                    1,
                )
                index += 1
            texts[i] = texts[i].replace("<|placeholder|>", "<|video_pad|>")
    return texts


def check_over_max_length(
    text: str,
    max_length: int,
    tokenizer: Qwen2Tokenizer,
    image_processor: Qwen2VLImageProcessor,
    image_grid_thw=None,
    video_grid_thw=None,
    replace_text: bool = True,
):
    if replace_text:
        text = replace_text_func(
            [text], image_processor, image_grid_thw, video_grid_thw
        )[0]
    input_ids = tokenizer(text)["input_ids"]
    if len(input_ids) > max_length:
        return True
    return False


#  qwen2vl的默认的padding方式是左padding...
def generate_labels(
    input_ids: torch.Tensor,
    texts: List[str],
    tokenizer: Qwen2Tokenizer,
    image_processor: Qwen2VLImageProcessor,
    template: Qwen2VLTemplate,
    replace_text: bool = True,
    image_grid_thw=None,
    video_grid_thw=None,
):
    """
    Args:
        input_ids: torch.Tensor, shape: [batch_size, seq_len]
        texts: List[str], shape: [batch_size]
        tokenizer: Qwen2Tokenizer
        image_processor: Qwen2VLImageProcessor
        template: Qwen2VLTemplate
        replace_text: bool,文本中如果含有图像，则需要将其替换为指定长度的占位符
        image_grid_thw: torch.Tensor, shape: [batch_size, 3]，和占位符计算有关
        video_grid_thw: torch.Tensor, shape: [batch_size, 3]，和占位符计算有关
    Returns:
        labels: torch.Tensor, shape: [batch_size, seq_len]
    """
    if replace_text:
        texts = replace_text_func(
            texts, image_processor, image_grid_thw, video_grid_thw
        )
    # # 做文本替换
    # if replace_text:
    #     if image_grid_thw is not None:
    #         merge_length = image_processor.merge_size**2
    #         index = 0
    #         for i in range(len(texts)):
    #             while "<|image_pad|>" in texts[i]:
    #                 texts[i] = texts[i].replace(
    #                     "<|image_pad|>",
    #                     "<|placeholder|>"
    #                     * (image_grid_thw[index].prod() // merge_length),
    #                     1,
    #                 )
    #                 index += 1
    #             texts[i] = texts[i].replace("<|placeholder|>", "<|image_pad|>")

    #     if video_grid_thw is not None:
    #         merge_length = image_processor.merge_size**2
    #         index = 0
    #         for i in range(len(texts)):
    #             while "<|video_pad|>" in texts[i]:
    #                 texts[i] = texts[i].replace(
    #                     "<|video_pad|>",
    #                     "<|placeholder|>"
    #                     * (video_grid_thw[index].prod() // merge_length),
    #                     1,
    #                 )
    #                 index += 1
    #             texts[i] = texts[i].replace("<|placeholder|>", "<|video_pad|>")

    # 生成labels
    targets = input_ids.clone()
    for text, target in zip(texts, targets):
        padding_len = int(target.eq(tokenizer.pad_token_id).sum())
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        target[:padding_len] = IGNORE_INDEX_ID
        cur_len = padding_len
        # 仅仅保留所有assistant的回答，其他的内容均使用IGNORE_INDEX进行填充
        # 使用assistant进行划分，除了第一部分之外，其他assitant之间的内容和可以按照
        # user区分，user后面的内容全部使用IGNORE_INDEX进行填充，user前面的内容保留
        # 最后可能会有assitant的响应的答案，特殊处理
        assistant_start = template.assistant_start
        parts = text.split(assistant_start)
        info = parts[0] + assistant_start
        temp_len = len(tokenizer(info)["input_ids"])
        target[cur_len : cur_len + temp_len] = IGNORE_INDEX_ID
        cur_len += temp_len

        for index in range(1, len(parts) - 1):
            info = parts[index]
            assistant_text, user_text = info.split(template.user_start)
            temp_len = len(tokenizer(assistant_text)["input_ids"])
            cur_len += temp_len
            part = template.user_start + user_text + template.assistant_start
            temp_len = len(tokenizer(part)["input_ids"])
            target[cur_len : cur_len + temp_len] = IGNORE_INDEX_ID
            cur_len += temp_len
        last_info = parts[-1]
        temp_len = len(tokenizer(last_info)["input_ids"])
        cur_len += temp_len

        if padding_len > 0:
            if total_len != cur_len - padding_len:
                target[:] = IGNORE_INDEX_ID
                print(
                    f"WARNING: tokenization mismatch: cur:{cur_len} vs. total:{total_len}."
                )
                sys.stdout.flush()

        return targets
