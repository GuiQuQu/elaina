"""
    SPDocVQA transform ocr data to layout text
"""

import json
import os
from typing import List, Dict
import math


class Point(object):
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y


class BBox(object):
    def __init__(self, boundingbox: List[int]) -> None:
        """
        boundingbox [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
        p1 是左上角, p2 是右上角, p3 是右下角, p4 是左下角
        (x1,y1) ..... (x2,y2)
        (x4,y4) ..... (x3,y3)
        """
        #
        x1, y1 = boundingbox[0:2]
        x2, y2 = boundingbox[2:4]
        x3, y3 = boundingbox[4:6]
        x4, y4 = boundingbox[6:8]
        self.p1 = Point(x1, y1)
        self.p2 = Point(x2, y2)
        self.p3 = Point(x3, y3)
        self.p4 = Point(x4, y4)
        self.width = max(x1, x2, x3, x4) - min(x1, x2, x3, x4)
        self.height = max(y1, y2, y3, y4) - min(y1, y2, y3, y4)
        self.top = min(y1, y2, y3, y4)
        self.bottom = max(y1, y2, y3, y4)
        self.left = min(x1, x2, x3, x4)
        self.right = max(x1, x2, x3, x4)


def in_the_same_line(bbox1: BBox, bbox2: BBox):
    """
    当两个BBox高度一致(height diff < 5)时,直接比较两个bbox.左上角高度是否一致(diff < 7)
    当两个BBox高度不一致时,尝试比较top,bottom,center对齐,只要有一个对齐,就认为在同一行
    """
    size_eps = 5
    position_eps = 7
    if abs(bbox1.height - bbox2.height) < size_eps:
        return abs(bbox1.p1.y - bbox2.p1.y) < position_eps
    else:
        # height is not the same, try top, bottom, center alignment,
        # as long as one is aligned, it is considered in the same line
        top_align = abs(bbox1.top - bbox2.top) < position_eps
        bottom_align = abs(bbox1.bottom - bbox2.bottom) < position_eps
        center_align = (
            abs(bbox1.top + bbox1.height / 2 - bbox2.top - bbox2.height / 2)
            < position_eps
        )
        return top_align or bottom_align or center_align


def sp_open_ocr_data(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)
    ocr_data = ocr_data["recognitionResults"][0]
    data = {
        "page_width": ocr_data["width"],
        "page_height": ocr_data["height"],
        "segments": [],
    }
    segments = []
    for seg in ocr_data["lines"]:
        segments.append({"text": seg["text"], "bbox": BBox(seg["boundingBox"])})
    data["segments"] = segments
    return data


def segments2lines(segments: list):
    """
    将segments划分成行
    """
    result: List[list] = []
    # 根据当前已有的lines的最后一个segment的bbox位置,
    # 判断当前seg是否可以在上述lines中确定和xxx同一行,如果可以,则添加到该行
    # 如果不可以,则新建一行
    for i, seg in enumerate(segments):
        add_old_lines = False
        for line in result:
            if in_the_same_line(line[-1]["bbox"], seg["bbox"]):
                line.append(seg)
                add_old_lines = True
                break
        if not add_old_lines:
            result.append([seg])

    # 从左到右排序同一行中所有的seg
    result = [sorted(line, key=lambda x: x["bbox"].left) for line in result]
    return result


def transform_ocr2layout(ocr_path, placeholder=" "):
    ocr_data = sp_open_ocr_data(ocr_path)
    if len(ocr_data["segments"]) == 0:
        return ""
    # 划分行
    ocr_data["lines"] = segments2lines(ocr_data["segments"])
    # 行内划分成segments和空格,来复制原本文档的布局
    page_width = ocr_data["page_width"]
    # 求解估计的每行最大字符数量
    max_estimate_line_char_cnt = estimate_line_char_cnt(ocr_data)
    max_estimate_unit_width_char_cnt:float = estimate_unit_width_char_cnt(ocr_data)
    # 单行拼接成文本并返回
    line_str_list = [
        line2text_by_unit_char_cnt(
            line,
            unit_width_char_cnt=max_estimate_unit_width_char_cnt,
            page_width=page_width,
            align="left",
            placeholder=placeholder,
        )
        for line in ocr_data["lines"]
    ]
    # 删除左右两侧的空格
    min_left_space = min([len(line) - len(line.lstrip()) for line in line_str_list])
    min_right_space = min([len(line) - len(line.rstrip()) for line in line_str_list])
    line_str_list = [line[min_left_space:-min_right_space] for line in line_str_list]
    document_str = "\n".join(line_str_list)
    return document_str


def estimate_unit_width_char_cnt(ocr_data):
    """
    求解最大的单位宽度占用的字符数量
    """
    # page_width = ocr_data["page_width"]
    max_estimate_unit_width_char_cnt = 0
    for line in ocr_data["lines"]:
        accum_width = 0
        accum_char_cnt = 0
        for seg in line:
            accum_width += seg["bbox"].width
            accum_char_cnt += len(seg["text"])
        # (accum_char_cnt/accum_width) * page_width
        estimate_cur_line_char_cnt = accum_char_cnt / accum_width
        max_estimate_unit_width_char_cnt = max(
            max_estimate_unit_width_char_cnt, estimate_cur_line_char_cnt
        )
    return max_estimate_unit_width_char_cnt


def estimate_line_char_cnt(ocr_data):
    """
    求解最大行宽度占用的字符数量
    """
    page_width = ocr_data["page_width"]
    max_estimate_line_char_cnt = 0
    for line in ocr_data["lines"]:
        accum_width = 0
        accum_char_cnt = 0
        for seg in line:
            accum_width += seg["bbox"].width
            accum_char_cnt += len(seg["text"])
        # (accum_char_cnt/accum_width) * page_width
        estimate_cur_line_char_cnt = math.ceil(
            accum_char_cnt * page_width / accum_width
        )
        max_estimate_line_char_cnt = max(
            max_estimate_line_char_cnt, estimate_cur_line_char_cnt
        )
    return max_estimate_line_char_cnt


def line2text_by_line_char_cnt(
    line: list, line_char_cnt: int, align="left", placeholder=" "
):
    pass
    if len(line) == 0:
        return ""
    line_str = ""
    for i, seg in enumerate(line):
        text:str = seg['text']
        bbox:BBox = seg['bbox']

def line2text_by_unit_char_cnt(
    line: list, unit_width_char_cnt: float, page_width, align="left", placeholder=" "
):
    """
    line : list, 每个元素是一个dict,包含text和bbox,表示一个segment
    unit_width_char_cnt : int, 单位宽度预估的估计的字符数量
    改好了, unit_width_char_cnt是一个浮点数,表示单位宽度占用的字符数量
    """

    if len(line) == 0:
        return ""
    ed = 0
    line_str = ""
    for i, seg in enumerate(line):
        text = seg["text"]
        bbox: BBox = seg["bbox"]
        # add left space
        left_space = math.ceil((bbox.left - ed) * unit_width_char_cnt)
        line_str += left_space * " "
        # add segments
        estimate_segment_char_cnt = math.ceil(bbox.width * unit_width_char_cnt)
        segment_str = get_segment_str(
            text, estimate_segment_char_cnt, align, placeholder
        )
        line_str += segment_str
        ed = bbox.right
    # add right space
    right_space = math.ceil((page_width - ed) * unit_width_char_cnt)
    line_str += right_space * " "
    return line_str


def get_segment_str(seg_text, segment_char_cnt, align="left", placeholder=" "):
    space_cnt = max(0, (segment_char_cnt - len(seg_text)))
    if align == "left":
        return seg_text + placeholder * space_cnt
    elif align == "right":
        return placeholder * space_cnt + seg_text
    elif align == "center":
        left_space = space_cnt // 2
        right_space = space_cnt - left_space
        return placeholder * left_space + seg_text + placeholder * right_space
    else:
        raise ValueError(f"align {align} is not supported")


def save_layout_result(save_path, document_str):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(document_str)

def test_transform_ocr2layout(ocr_path):
    document_str = transform_ocr2layout(ocr_path, placeholder=" ")
    save_path = ocr_path.replace(".json", "_layout.txt")
    save_layout_result(save_path=save_path, document_str=document_str)

if __name__ == "__main__":
    ocr_path = "/root/elaina/examples/sp/ffbf0023_4.json"
    test_transform_ocr2layout(ocr_path)
    ocr_path = "/root/elaina/examples/sp/ffbf0227_1.json"
    test_transform_ocr2layout(ocr_path)
    ocr_path = "/root/elaina/examples/sp/ffbx0227_7.json"
    test_transform_ocr2layout(ocr_path)
    ocr_path = "/root/elaina/examples/sp/ffdh0224_1.json"
    test_transform_ocr2layout(ocr_path)

