from typing import List, Tuple
import json
from collections import defaultdict
from transformers import PreTrainedTokenizer


def truncate_layout(
    layout: str, tokenizer: PreTrainedTokenizer = None, max_token_length: int = 1024
) -> Tuple[str, bool]:
    """
    truncate layout to fit the max_token_length
    return truncated layout and is_truncated(True or False)
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
    return "\n".join(reserved_lines), is_truncated

def truncate_layout_by_length(
        layout: str, tokenizer: PreTrainedTokenizer = None, max_token_length: int = 1024
) -> Tuple[str, bool]:
    """
        强制按照max_token_length截断layout
    """
    if tokenizer == None:
        return layout
    is_truncated = False
    
    if len(layout) == 0:
        return layout, is_truncated
    
    layout_input_ids = tokenizer(layout, return_tensors="pt").input_ids
    layout_input_ids = layout_input_ids.squeeze(0)
    if layout_input_ids.size(0) > max_token_length:
        is_truncated = True
    layout_input_ids = layout_input_ids[:max_token_length]
    layout = tokenizer.decode(layout_input_ids)
    return layout, is_truncated


def groupby_classify_result(classify_result_path, model_output_key = 'model_output'):
        """
            Args:
                classify_result_path: str, path to classify result json file
                model_output_key: str, key of model output in classify result json file
            Returns:
                qid2items: {qid: dict({page_id: model_output_score})}
        """
        qid2items = defaultdict(dict)
        with open(classify_result_path, 'r', encoding='utf-8') as f:
            classify_result = json.load(f)
        for item in classify_result:
            qid = item['qid']
            page_id = item['image_path'].split('/')[-1].split('.')[0]
            qid2items[qid][page_id] = item[model_output_key]

        return qid2items

