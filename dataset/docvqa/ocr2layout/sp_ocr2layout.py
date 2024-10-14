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


def transform_ocr2layout(ocr_path):
    ocr_data = sp_open_ocr_data(ocr_path)
    if len(ocr_data["segments"]) == 0:
        return ""

    # 划分行
    ocr_data["lines"] = segments2lines(ocr_data["segments"])
    # 行内划分成segments和空格,来复制原本文档的布局
    page_width = ocr_data["page_width"]
    # 求解估计的每行最大字符数量
    max_estimate_line_char_cnt = estimate_line_char_cnt(ocr_data)
    # 拼接成文本并返回


def estimate_line_char_cnt(ocr_data):
    """
        求解单位宽度占用的字符数量
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
        max_estimate_line_char_cnt = max(max_estimate_line_char_cnt, estimate_cur_line_char_cnt)
    return max_estimate_line_char_cnt


def line2text(line: list, line_char_cnt: int):
    """
    line : list, 每个元素是一个dict,包含text和bbox,表示一个segment
    line_width : int, 行的宽度,字符数量
    """
    ed = 0
    for _, seg in enumerate(line):
        text = seg['text']
        bbox: BBox = seg['bbox']
        # add left space
        left_space = (bbox.left - ed)
