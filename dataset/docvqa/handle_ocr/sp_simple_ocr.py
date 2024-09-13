import json
import math

"""
    lines : List[Dict] List of text segment
    text segment : Dict[
    boundingBox,
    text,
    words[Dict[boundingBox,text]]
"""
def sp_open_ocr_data(json_path: str):
    with open(json_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data["recognitionResults"][0]

# boundingbox [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
#             [(622,138),(1137,136),(1138,167),(622,168)]
# 分别指定左上角,右上角,右下角,左下角


def sp_get_layout(ocr_data: dict, placeholder: str = " ") -> str:
    # step 1. get line for ocr_data
    page_width = ocr_data["width"]
    # page_height = ocr_data["height"]
    all_lines = get_lines_from_ocr_data(ocr_data)
    # cnt_star指文档中最大的行字符数量
    cnt_star = 0
    # star_text = ""
    for line in all_lines:
        for segment in line:
            box = parse_boundingbox(segment["boundingBox"])
            char_cnt = len(segment["text"])
            # 要求width至少为char_cnt
            width = max(box["width"],char_cnt)
            t = math.ceil(char_cnt * page_width / width)
            if cnt_star < t:
                cnt_star = t
                star_text = segment['text']
            # cnt_star = max(cnt_star, math.ceil(char_cnt * page_width / width))
    # print(f"cnt_star:{cnt_star}, text:{star_text}")
    # step 2. add space for every line
    res = []
    for line in all_lines:
        line_str = ""
        ed = 0
        for i in range(len(line)):
            segment = line[i]
            box = parse_boundingbox(segment["boundingBox"])
            left = box["left"]
            width = box["width"]
            # add space
            left_space_cnt = math.ceil((left - ed) / page_width * cnt_star)
            line_str += " " * left_space_cnt
            # add text
            text_cnt = math.ceil(width / page_width * cnt_star)
            text = segment["text"] + placeholder * (text_cnt - len(segment["text"]))
            line_str += text
            ed = left + width
        # add last right space
        right_space_cnt = cnt_star - len(line_str)
        line_str += " " * right_space_cnt
        res.append(line_str)
    # 经过检测，发现数据中存在两个完全没有识别出ocr文本的情况
    if len(res) == 0: return ""
    # 清除左右两侧的空格
    min_left_space_cnt = min([len(line) - len(line.lstrip()) for line in res])
    min_right_space_cnt = min([len(line) - len(line.rstrip()) for line in res])
    for i in range(len(res)):
        res[i] = res[i][min_left_space_cnt:len(res[i]) - min_right_space_cnt]
    res = "\n".join(res)
    # with open("layout.txt", "w", encoding="utf-8") as f:
    #     f.write(res)
    return res


def parse_boundingbox(boundingbox) -> dict:
    """
        解析boundingbox,返回left, top, width, height
    """
    x1, y1 = boundingbox[0:2]
    x2, y2 = boundingbox[2:4]
    x3, y3 = boundingbox[4:6]
    x4, y4 = boundingbox[6:8]

    # 因为可能存在误差,导致width or height == 0 的情况,这做简单的修正
    ret = dict(
        width=(x2 - x1 + x3 - x4) / 2,
        height=(y4 - y1 + y3 - y2) / 2,
        left=(x1+x4) / 2,
        right=(x2+x3) / 2,
        top=(y1+y2) / 2,
        bottom=(y4+y3) / 2,
    )
    # if ret["width"] == 0: ret["width"] += 1
    # if ret["height"] == 0: ret["height"] += 1
    return ret


def get_lines_from_ocr_data(ocr_data: dict) -> list:

    def is_in_the_same_line(pre, cur) -> bool:
        """
            判定上一个词和当前词是否在同一行
        """
        size_eps = 5
        position_eps = 7
        pre_box = parse_boundingbox(pre["boundingBox"])
        cur_box = parse_boundingbox(cur["boundingBox"])
        pre_height, pre_top, pre_bottom = pre_box["height"], pre_box["top"], pre_box["bottom"]
        cur_height, cur_top, cur_bottom = cur_box["height"], cur_box["top"], cur_box["bottom"]
        if abs(pre_height - cur_height) < size_eps:
            return abs(pre_top - cur_top) < position_eps
        else:
            # 字体大小不一致,尝试上,中,下对齐,只要有一个对齐就可以
            top_align = abs(pre_top - cur_top) < position_eps
            bottom_align = abs(pre_bottom - cur_bottom) < position_eps
            center_align = abs(pre_top + pre_height / 2 -
                               cur_top - cur_height / 2) < position_eps
            return top_align or bottom_align or center_align

    text_segments = ocr_data["lines"]
    all_lines = []
    for i, text_segment in enumerate(text_segments):
        if i == 0:
            all_lines.append([text_segment])
        else:
            # if width == 0 or height == 0: continue;
            box = parse_boundingbox(text_segment["boundingBox"])
            # 忽略测量宽度和高度为0的box(误差)
            if box["width"] == 0 or box["height"] == 0: continue
            # 忽略所有倾斜的or竖直的text_segment
            if box["width"] < len(text_segment['text']): continue
            cur = text_segment
            add_success = False
            for line in all_lines:
                if is_in_the_same_line(line[-1], cur):
                    line.append(cur)
                    add_success = True
                    break
            if not add_success:
                all_lines.append([cur])
    # sort every lines by left
    all_lines = [sorted(line, key=lambda x: parse_boundingbox(
        x["boundingBox"])["left"]) for line in all_lines]
    return all_lines


def print_layout(layout: str):
    lines = layout.split("\n")
    max_char_cnt = max([len(line) for line in lines])
    lines.insert(0, "-" * (max_char_cnt))
    for i, line in enumerate(lines):
        lines[i] = "| " + line + " |"
    lines.append("-" * (max_char_cnt+4))
    print("\n".join(lines))


def print_lines(all_lines: list):
    for i, line in enumerate(all_lines):
        print(f"line {i}:")
        for segment in line:
            print(segment["text"], end="------")
        print("\n")

def sp_get_layout_by_json_path(json_path:str, placeholder:str = " "):
    ocr_data = sp_open_ocr_data(json_path)
    return sp_get_layout(ocr_data,placeholder)

def sp_get_baseline_layout_by_json_path(json_path:str):
    """
        baseline 是仅仅将text segment 拼在一起
    """
    ocr_data = sp_open_ocr_data(json_path)
    text_segments = ocr_data["lines"]
    text_segments = [t["text"] for t in text_segments]
    return " ".join(text_segments)

def sp_get_lines_layout_by_json_path(json_path:str):
    """
        该做法仅仅划分行
    """
    ocr_data = sp_open_ocr_data(json_path)
    all_lines = get_lines_from_ocr_data(ocr_data)
    all_lines = [" ".join([t["text"] for t in line]) 
                 for line in all_lines]
    return "\n".join(all_lines)


def save_ocr_result(json_dir,ocr_dir):
    import os
    json_files = os.listdir(json_dir)
    for json_file in json_files:
        json_path = os.path.join(json_dir,json_file)
        layout = sp_get_layout_by_json_path(json_path, placeholder="*")
        ocr_file = json_file.replace(".json",".txt")
        ocr_path = os.path.join(ocr_dir,ocr_file)
        with open(ocr_path,"w",encoding="utf-8") as f:
            f.write(layout)
        print(f"save {ocr_path} success")

def main():
    # 这两个json文件中没有ocr识别结果
    # /home/klwang/data/SPDocVQA/ocr/jzhd0227_85.json 
    # /home/klwang/data/SPDocVQA/ocr/hpbl0226_5.json
    json_path = "/home/klwang/code/GuiQuQu-docvqa-vllm-inference/src/handle_ocr/sp/ffbf0023_4.json"
    image_path = "/home/klwang/data/SPDocVQA/images/hpbl0226_5.png"
    layout1 = sp_get_baseline_layout_by_json_path(json_path)
    layout2 = sp_get_lines_layout_by_json_path(json_path)
    layout3 = sp_get_layout_by_json_path(json_path,placeholder="*")

    with open("layout.txt","w",encoding="utf-8") as f:
        f.write(layout3)
    # print(len(layout))
    # print(layout)


if __name__ == "__main__":
    main()
