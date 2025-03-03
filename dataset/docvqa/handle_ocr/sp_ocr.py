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


def _split_line_from_data(data: List[dict]):
    """
    split line from data
    """
    result = []
    for i, seg in enumerate(data):
        if i == 0:
            result.append([seg])
        else:
            # test cur segment with every line last segment,
            # try add cur segment
            # try success, add to that line, else add to new line
            add_success = False
            for l in result:
                if in_the_same_line(l[-1]["bbox"], seg["bbox"]):
                    l.append(seg)
                    add_success = True
                    break
            if not add_success:
                result.append([seg])

    # sort bbox from left to right
    result = [sorted(line, key=lambda x: x["bbox"].left) for line in result]
    return result


def _sp_layout_no_placeholder(data):
    """
    from 'Layout and Task Aware Instruction Prompt for Zero-shot Document Image Question Answering'
    self implemention
    """
    max_line_char_cnt = 0  # 字符数最多的一行
    max_line_width = 0  # 该行对应的宽度
    document = data["lines"]
    for line in document:
        line_char_cnt = 0
        line_text_width = 0.0
        for seg in line:
            line_char_cnt += len(seg["text"])
            line_text_width += seg["bbox"].width
        if max_line_char_cnt < line_char_cnt:
            max_line_char_cnt = line_char_cnt
            max_line_width = line_text_width
    document_lines = []
    for line in document:
        line_str = ""
        for i, seg in enumerate(line):
            text: str = seg["text"]
            bbox: BBox = seg["bbox"]
            if i == 0:
                line_str += text
            else:
                space_cnt = math.ceil(
                    (bbox.left - line[i - 1]["bbox"].right)
                    / max_line_width
                    * max_line_char_cnt
                )
                line_str = line_str + " " * space_cnt + text
        document_lines.append(line_str)
    if len(document_lines) == 0:
        return ""
    document_str = "\n".join(document_lines)
    return document_str


def _sp_layout_with_placeholder(data, placeholder="*"):
    max_line_char_cnt = 0
    page_width = data["page_width"]
    document = data["lines"]
    for line in document:
        line_char_width = 0.0
        line_char_cnt = 0
        for seg in line:
            line_char_cnt += len(seg["text"])
            line_char_width += seg["bbox"].width
        t = math.ceil(line_char_cnt / line_char_width * page_width)
        max_line_char_cnt = max(max_line_char_cnt, t)

    document_lines = []
    for line in document:
        line_str = ""
        ed = 0
        for _, seg in enumerate(line):
            text: str = seg["text"]
            bbox: BBox = seg["bbox"]
            # add space
            left_space = math.ceil(max_line_char_cnt * (bbox.left - ed) / page_width)
            left_space = " " * left_space
            line_str += left_space
            # add text
            expected_text_cnt = math.ceil(max_line_char_cnt * (bbox.width / page_width))
            text = text + placeholder * (expected_text_cnt - len(text))
            line_str += text
            ed = bbox.right
        # add right space
        right_space = max_line_char_cnt - len(line_str)
        line_str += " " * right_space
        document_lines.append(line_str)

    if len(document_lines) == 0:
        return ""

    # strip the left and right space
    min_left_space_cnt = min(
        [len(line) - len(line.lstrip()) for line in document_lines]
    )
    min_right_space_cnt = min(
        [len(line) - len(line.rstrip()) for line in document_lines]
    )
    document_lines = [
        line[min_left_space_cnt : len(line) - min_right_space_cnt]
        for line in document_lines
    ]
    document_str = "\n".join(document_lines)
    return document_str


def sp_layout_no_placeholder_from_json_path(json_path: str):
    "From LATIN-Prompt"
    data = sp_open_ocr_data(json_path)
    data["lines"] = _split_line_from_data(data["segments"])
    return _sp_layout_no_placeholder(data)


def sp_layout_star_from_json_path(json_path: str):
    "Layout Text, placeholder is *"
    data = sp_open_ocr_data(json_path)
    data["lines"] = _split_line_from_data(data["segments"])
    return _sp_layout_with_placeholder(data, placeholder="*")


def sp_layout_space_from_json_path(json_path: str):
    "Layout Text, placeholder is space"
    data = sp_open_ocr_data(json_path)
    data["lines"] = _split_line_from_data(data["segments"])
    return _sp_layout_with_placeholder(data, placeholder=" ")


def sp_layout_lines_from_json_path(json_path: str):
    "Line Text, is 'lines'"
    data = sp_open_ocr_data(json_path)
    data["lines"] = _split_line_from_data(data["segments"])
    document = [" ".join([seg["text"] for seg in line]) for line in data["lines"]]
    document_str = "\n".join(document)
    return document_str


def sp_layout_no_handle_from_json_path(json_path: str):
    "No Any Handle, is 'words'"
    data = sp_open_ocr_data(json_path)
    document = [seg["text"] for seg in data["segments"]]
    document_str = " ".join(document)
    return document_str


def generate_docvqa_layout(ocr_dir, layout_dir):
    """
    generate docvqa layout from ocr data,save to layout_dir
    """
    if not os.path.exists(layout_dir):
        os.makedirs(layout_dir)
    for file in os.listdir(ocr_dir):
        if file.endswith(".json"):
            json_path = os.path.join(ocr_dir, file)
            layout = sp_layout_star_from_json_path(json_path)
            layout_path = os.path.join(layout_dir, file.replace(".json", ".txt"))
            with open(layout_path, "w", encoding="utf-8") as f:
                f.write(layout)


def test_star_layout():
    json_path = "/home/klwang/code/GuiQuQu-docvqa-vllm-inference/src/handle_ocr/sp/jmlh0227_8.json"
    sp_layout = sp_layout_star_from_json_path(json_path)
    with open("layout_sp_star_.txt", "w", encoding="utf-8") as f:
        f.write(sp_layout)


def test_no_placeholder_layout():
    json_path = "/home/klwang/code/GuiQuQu-docvqa-vllm-inference/src/handle_ocr/sp/jmlh0227_8.json"
    sp_layout = sp_layout_no_placeholder_from_json_path(json_path)
    with open("layout_sp_np.txt", "w", encoding="utf-8") as f:
        f.write(sp_layout)


def main():
    test_no_placeholder_layout()
    test_star_layout()


if __name__ == "__main__":
    main()
