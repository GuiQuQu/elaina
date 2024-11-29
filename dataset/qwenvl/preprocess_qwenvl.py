from typing import List
import sys
import torch
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_INDEX_ID = LabelSmoother.ignore_index

from models.docvqa.qwenvl.tokenization_qwen import QWenTokenizer
from dataset.qwenvl.template import QwenVLTemplate


def generate_labels_for_qwenvl(
    input_ids: torch.Tensor,
    texts: List[str],
    tokenizer: QWenTokenizer,
    template: QwenVLTemplate,
) -> torch.Tensor:
    # 生成labels
    targets = input_ids.clone()
    pad_id = tokenizer.eod_id
    for text, target in zip(texts, targets):
        total_len = int(target.ne(pad_id).sum())
        cur_len = 0
        # 仅仅保留所有assistant的回答，其他的内容均使用IGNORE_INDEX进行填充
        # 使用assistant进行划分，除了第一部分之外，其他assitant之间的内容和可以按照
        # user区分，user后面的内容全部使用IGNORE_INDEX进行填充，user前面的内容保留
        # 最后可能会有assitant的响应的答案，特殊处理

        # qwenvl在处理labels的时候，会保留所有<|im_start|>, <|im_end|>\n 组合对应的label

        assistant_start = template.assistant_start  # <|im_start|>assistant\n
        parts = text.split(assistant_start)
        # part[0] 可能存在system message
        info = parts[0]
        system_message, user_text = info.split(template.user_start)
        # encode system message
        system_message = system_message[len(template.im_start) : -len(template.im_end+template.sep)]
        cur_len += 1 # <|im_start|>
        temp_len = len(tokenizer(system_message)["input_ids"])
        target[cur_len : cur_len + temp_len] = IGNORE_INDEX_ID
        cur_len += temp_len
        cur_len += 2 # <|im_end|>\n

        # encode user text 

        cur_len += 1 # <|im_start|>
        # user\nuser_text
        part = template.user_start[len(template.im_start):] + user_text[:-len(template.im_end+template.sep)]
        temp_len = len(tokenizer(part)["input_ids"])
        target[cur_len : cur_len + temp_len] = IGNORE_INDEX_ID
        cur_len += temp_len
        cur_len += 2 # <|im_end|>\n

        # encode assistant start
        cur_len += 1 # <|im_start|>
        # assistant\n
        temp_len = len(tokenizer(template.assistant_start[len(template.im_start):])["input_ids"])
        target[cur_len : cur_len + temp_len] = IGNORE_INDEX_ID
        cur_len += temp_len

        for index in range(1, len(parts) - 1):
            info = parts[index]
            assistant_text, user_text = info.split(template.user_start)

            # assistant 部分
            temp_len = len(tokenizer(assistant_text)["input_ids"])
            cur_len += temp_len
            
            # user 部分
            cur_len += 1 # <|im_start|>  
            # user\n:user_text<|im_end|>\n
            part = template.user_start[len(template.im_start):] + user_text[:-len(template.im_end+template.sep)]
            temp_len = len(tokenizer(part)["input_ids"])
            target[cur_len : cur_len + temp_len] = IGNORE_INDEX_ID
            cur_len += temp_len
            cur_len += 2 # <|im_end|>\n

            cur_len += 1 # <|im_start|>
            # assistant\n
            temp_len = len(tokenizer(template.assistant_start[len(template.im_start):])["input_ids"])
            target[cur_len : cur_len + temp_len] = IGNORE_INDEX_ID
            cur_len += temp_len
        
        last_info = parts[-1]
        temp_len = len(tokenizer(last_info)["input_ids"])
        cur_len += temp_len

        # 处理padding部分
        target[cur_len:] = IGNORE_INDEX_ID
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX_ID
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                )
                sys.stdout.flush()

    return targets
