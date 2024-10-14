from typing import List, Tuple
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
