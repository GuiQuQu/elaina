import json
import math
from typing import List,Dict
import logging

logger = logging.getLogger(__name__)

class MPBBox(object):
    def __init__(self,bbox_dict):
        self.width = bbox_dict["Width"]
        self.height = bbox_dict["Height"]
        self.top = bbox_dict["Top"]
        self.left = bbox_dict["Left"]
        self.bottom = self.top + self.height
        self.right = self.left + self.width

def mp_open_ocr_data(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ocr_data = {}
    bbox = MPBBox(data["PAGE"][0]["Geometry"]["BoundingBox"])
    ocr_data["page_width"] = bbox.width
    ocr_data["page_height"] = bbox.height
    segments = []
    for line in data.get("LINE",[]):
        item = {}
        item['text'] = line["Text"]
        item['bbox'] = MPBBox(line["Geometry"]["BoundingBox"])
        item['confidence'] = line["Confidence"]
        segments.append(item)
    if segments == []:
        logger.warning("No OCR data in %s", json_path)
    ocr_data["segments"] = segments
    return ocr_data

def is_same_line(bbox1:MPBBox,bbox2:MPBBox):
    size_eps = 0.006
    position_eps = 0.006
    if (abs(bbox1.height - bbox2.height) < size_eps):
        return abs(bbox1.top - bbox2.top) < position_eps
    else:
        # height is not the same, try top, bottom, center alignment, 
        # as long as one is aligned, it is considered in the same line
        top_align = abs(bbox1.top - bbox2.top) < position_eps
        bottom_align = abs(bbox1.bottom - bbox2.bottom) < position_eps
        center_align = abs(bbox1.top + bbox1.height / 2 - bbox2.top - bbox2.height / 2) < position_eps
        return top_align or bottom_align or center_align

def _split_line_from_data(data:List[dict]):
    pass
    result = []
    for i, seg in enumerate(data):
        if i == 0:
            result.append([seg])
        else:
            add_success = False
            for l in result:
                if is_same_line(l[-1]['bbox'],seg['bbox']):
                    l.append(seg)
                    add_success = True
                    break
            if not add_success:
                result.append([seg])
    # sort bbox from left to right
    result = [sorted(line,key=lambda x:x['bbox'].left) for line in result]
    return result

def _mp_layout_with_placeholder(data:Dict[str,any],placeholder:str|None):
    "placeholder: str"
    max_line_char_cnt = 0
    page_width = data['page_width']
    document = data['lines']
    for line in document:
        line_char_cnt = 0
        line_text_width = 0.0
        for seg in line:
            line_char_cnt += len(seg['text'])
            line_text_width += seg['bbox'].width
        t = math.ceil((page_width / line_text_width) * line_char_cnt)
    max_line_char_cnt = max(max_line_char_cnt,t)

    document_lines:List[str] = []
    for line in document:
        line_str = ""
        ed = 0
        for i, seg in enumerate(line):
            text:str = seg['text']
            bbox:MPBBox = seg['bbox']
            if placeholder == None:
                line_str = line_str + text + " "
                if i == len(line) - 1: line_str = line_str[:-1]
            else:
                # add space
                left_space_cnt = math.ceil((bbox.left - ed) * max_line_char_cnt)
                line_str = line_str + " " * left_space_cnt
                # add text
                excepted_text_cnt = math.ceil(bbox.width * max_line_char_cnt)
                text = text + placeholder * (excepted_text_cnt - len(text))
                line_str = line_str + text
                ed = bbox.right
        # add last right space
        right_space_cnt = max_line_char_cnt - len(line_str)
        line_str = line_str + " " * right_space_cnt
        document_lines.append(line_str)

    if len(document_lines) == 0:
        return ""
    # strip the left and right space
    min_left_space_cnt = min([len(line) - len(line.lstrip()) for line in document_lines])
    min_right_space_cnt = min([len(line) - len(line.rstrip()) for line in document_lines])
    document_lines = [line[min_left_space_cnt:len(line) - min_right_space_cnt] for line in document_lines]
    document_str = "\n".join(document_lines)
    return document_str

def mp_laytout_from_json_path(json_path, placeholder= '*'):
    data = mp_open_ocr_data(json_path)
    if len(data['segments']) == 0:
        return ""
    data['lines'] = _split_line_from_data(data['segments'])
    return _mp_layout_with_placeholder(data,placeholder)

def main():
    json_path = "/home/klwang/code/GuiQuQu-docvqa-vllm-inference/src/handle_ocr/mp/snbx0223_p19.json"
    mp_layout = mp_laytout_from_json_path(json_path, placeholder="*")
    with open("./mp/layout_mp.txt","w",encoding="utf-8") as f:
        f.write(mp_layout)

if __name__ == "__main__":
    main()