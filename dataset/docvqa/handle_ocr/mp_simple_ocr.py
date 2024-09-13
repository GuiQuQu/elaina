"""
    按照ocr的识别结果,使用str复刻layout
    notice
    1. 处理识别结果时,按照text segment的划分来处理,不要按照word来处理
"""

import json
import math


def mp_open_ocr_data(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def mp_get_layout(ocr_data: dict) -> str:
    # step 1. get line for ocr_data
    all_lines = get_lines_from_ocr_data(ocr_data)
    # cnt_star指文档中最大的行字符数量
    cnt_star = 0
    for line in all_lines:
        for segment in line:
            width = segment["Geometry"]["BoundingBox"]["Width"]
            char_cnt = len(segment["Text"])
            cnt_star = max(cnt_star, math.ceil(char_cnt / width))
    # step 2. add space for every line
    res = []
    for line in all_lines:
        line_str = ""
        ed = 0
        for i in range(len(line)):
            segment = line[i]
            left = segment["Geometry"]["BoundingBox"]["Left"]
            width = segment["Geometry"]["BoundingBox"]["Width"]
            # add space
            left_space_cnt = math.ceil((left - ed) * cnt_star)
            line_str += " " * left_space_cnt
            # add text
            text_cnt = math.ceil(width * cnt_star)
            text = segment["Text"] + "*" * (text_cnt - len(segment["Text"]))
            line_str += text
            ed = left + width
        # add last right space
        right_space_cnt = cnt_star - len(line_str)
        line_str += " " * right_space_cnt
        res.append(line_str)
    # 清除左右两侧的空格
    min_left_space_cnt = min([len(line) - len(line.lstrip()) for line in res])
    min_right_space_cnt = min([len(line) - len(line.rstrip()) for line in res])
    for i in range(len(res)):
        res[i] = res[i][min_left_space_cnt : len(res[i]) - min_right_space_cnt]
    return "\n".join(res)


def get_lines_from_ocr_data(ocr_data: dict) -> list:

    def is_in_the_same_line(pre, cur) -> bool:
        """
        判定上一个词和当前词是否在同一行
        """
        size_eps = 0.005
        position_eps = 0.005
        pre_height = pre["Geometry"]["BoundingBox"]["Height"]
        cur_height = cur["Geometry"]["BoundingBox"]["Height"]
        pre_top = pre["Geometry"]["BoundingBox"]["Top"]
        cur_top = cur["Geometry"]["BoundingBox"]["Top"]
        pre_bottom = pre["Geometry"]["BoundingBox"]["Top"] + pre_height
        cur_bottom = cur["Geometry"]["BoundingBox"]["Top"] + cur_height
        if abs(cur_height - pre_height) < size_eps:
            # 字体大小一致,采用上对齐方式
            return abs(cur_top - pre_top) < position_eps
        else:
            # 字体大小不一致,尝试上,中,下对齐,只要有一个可以对齐就可以
            top_align = abs(cur_top - pre_top) < position_eps
            center_align = (
                abs(cur_top + cur_height / 2 - pre_top - pre_height / 2) < position_eps
            )
            bottm_align = abs(cur_bottom - pre_bottom) < position_eps
            return top_align or center_align or bottm_align

    text_segments = ocr_data["LINE"]
    all_lines = []
    for i, text_segment in enumerate(text_segments):
        if i == 0:
            all_lines.append([text_segment])
        else:
            pre = all_lines[-1][-1]
            cur = text_segment
            if is_in_the_same_line(pre, cur):
                all_lines[-1].append(cur)
            else:
                all_lines.append([cur])
    return all_lines


def print_lines(all_lines: list):
    for i, line in enumerate(all_lines):
        print(f"line {i}:")
        for word in line:
            print(word["Text"], end="------")
        print("\n")


def print_layout(layout: str):
    layout = get_layout_show_string(layout)
    print(layout)


def get_layout_show_string(layout: str):
    lines = layout.split("\n")
    max_char_cnt = max([len(line) for line in lines])
    lines.insert(0, "-" * (max_char_cnt))
    for i, line in enumerate(lines):
        lines[i] = "| " + line + " |"
    lines.append("-" * (max_char_cnt + 4))
    return "\n".join(lines)


def main():
    json_path = "./ffdw0217_p8.json"
    ocr_data = mp_open_ocr_data(json_path)
    layout = mp_get_layout(ocr_data)
    print(len(layout))
    print(layout)
    # print_layout(layout)


if __name__ == "__main__":
    main()
