
from functools import partial
from typing import List,Tuple
from transformers import PreTrainedTokenizer
import json
import random
import numpy as np
import torch

import dataset.docvqa.handle_ocr as handle_ocr

def truncate_layout(layout:str, 
                    tokenizer:PreTrainedTokenizer = None, 
                    max_token_length:int = 1024) -> str:
    """
        truncate layout to fit the max_token_length
        another version of truncate_layout2, return truncated layout and is_truncated(True or False)
    """
    if tokenizer == None:
        return layout
    lines = layout.split("\n")
    lines_input_ids = [tokenizer([l], return_tensors="pt").input_ids for l in lines]
    reserved_lines = []
    ids_cnt = 0
    is_truncated = False
    for i, input_ids in enumerate(lines_input_ids):
        if ids_cnt + input_ids.size(-1) < max_token_length:
            ids_cnt += input_ids.size(-1)
            reserved_lines.append(lines[i])
        else: 
            is_truncated = True
            break
    return "\n".join(reserved_lines)

def truncate_layout2(layout:str, 
                    tokenizer:PreTrainedTokenizer = None, 
                    max_token_length:int = 1024) -> Tuple[str, bool]:
    """
        truncate layout to fit the max_token_length
        return truncated layout and is_truncated(True or False)
    """
    if tokenizer == None:
        return layout
    lines = layout.split("\n")
    lines_input_ids = [tokenizer([l],return_tensors='pt').input_ids for l in lines]
    reserved_lines = []
    ids_cnt = 0
    is_truncated = False
    for i, input_ids in enumerate(lines_input_ids):
        if ids_cnt + input_ids.size(-1) < max_token_length:
            ids_cnt += input_ids.size(-1)
            reserved_lines.append(lines[i])
        else: 
            is_truncated = True
            break
    return "\n".join(reserved_lines), is_truncated


def open_json(json_path: str):
    """
        打开一个json文件
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def load_data(json_path: str):
    """
        打开json文件,并返回其中的data字段value内容
        适用于spdocvqa的数据加载
    """
    data = open_json(json_path)
    return data["data"]

def seed_everything(seed):
    """
        seed_everything
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_layout_func(type:str):
    """
        根据给定的type返回对应的layout处理方式(目前只处理sp)
    """
    if type == "all-star":
        return partial(handle_ocr.sp_get_layout_by_json_path, placeholder="*")
    elif type == "lines":
        return handle_ocr.sp_get_lines_layout_by_json_path
    elif type == "words":
        return handle_ocr.sp_get_baseline_layout_by_json_path
    elif type == "none":
        return None
    else:
        raise ValueError("Not support layout pattern")

def sp_get_layout_func2(type:str):
    """
        基于新写的sp layout获取代码获取layout
        1. 不会跳过任何ocr segment
        2. 增加sp_layout_no_placeholder_from_json_path
    """
    if type == "all-star":
        return handle_ocr.sp_layout_star_from_json_path
    elif type == "all-space":
        return handle_ocr.sp_layout_space_from_json_path
    elif type == "lines":
        return handle_ocr.sp_layout_lines_from_json_path # 仅做行切分
    elif type == "words":
        return handle_ocr.sp_layout_no_handle_from_json_path # 把所有的text segment仅用空格隔开
    elif type == "no-placeholder":
        return handle_ocr.sp_layout_no_placeholder_from_json_path # LATIN-Prompt的个人实现
    else:
        raise ValueError("Not support layout pattern")

def mp_get_layout_func(type:str):
    """
        mp docvqa layout获取函数
    """
    if type == "all-star":
        return partial(handle_ocr.mp_laytout_from_json_path,placeholder="*")
    elif type == 'all-space':
        return partial(handle_ocr.mp_laytout_from_json_path,placeholder=" ")
    else:
        raise ValueError(f"Not support layout pattern: {type}")

class Response(object):
    def __init__(self, text:str, 
                 prompt:str,
                 end_reason:str,
                 input_ids:List[int], 
                 output_ids:List[int]):
        self.text = text
        self.prompt = prompt
        self.end_reason = end_reason
        self.input_ids = input_ids
        self.output_ids = output_ids

    
    def __str__(self) -> str:
        return f"Response(text={self.text}, prompt={self.prompt}, input_ids={self.input_ids}, output_ids={self.output_ids})"