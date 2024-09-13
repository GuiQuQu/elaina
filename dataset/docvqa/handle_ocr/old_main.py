"""
    弃用
    WORKDIR is `pwd`
"""
import json
import math


def open_json(json_path: str):
    with open(json_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_all_lines(ocr_data: dict) -> str:
    lines = ocr_data["LINE"]
    all_lines = []
    for i, lines in enumerate(lines):
        all_lines.append(lines["Text"])
    return "\n".join(all_lines)


"""
    处理字体大小相同的内容很好处理,无论是采用top,center,bottom,
    都可以得到一致的结果,但是对于字体大小不一致内容,则会出现了,这些问题
    既有可能上对齐,也有可能下对齐,也有可能中对齐
    
    字体大小一致,采用上对齐方式
    字体大小不一致,尝试上,中,下对齐,只要有一个可以对齐就可以
"""


def get_lines_by_words(ocr_data: dict) -> str:
    words = ocr_data["WORD"]
    handwritting_eps, printed_eps, size_eps = 0.2, 0.1, 0.005
    position_eps = 0.005

    def check_in_same_line(pre, cur):
        """
            判定上一个词和当前词是否在同一行
        """
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
            center_align = abs(cur_top + cur_height / 2 -
                               pre_top - pre_height / 2) < position_eps
            bottm_align = abs(cur_bottom - pre_bottom) < position_eps
            return top_align or center_align or bottm_align

    all_lines = []
    cur_line = [words[0]]
    last_word = words[0]

    for i in range(1, len(words)):
        word = words[i]
        if check_in_same_line(last_word, word):
            cur_line.append(word)
        else:
            all_lines.append(cur_line)
            cur_line = [word]
        last_word = word
    all_lines.append(cur_line)

    return get_article_by_words_lines(all_lines)


def get_article_by_words_lines(lines: list) -> str:
    """
        lines : list[word(dict)]
        len(lines) is the number of lines
    """
    # sort by left start
    lines = [sorted(words, key=lambda x:  x["Geometry"]
                    ["BoundingBox"]["Left"]) for words in lines]
    # 预计一行中字符的数量,统计每一行
    line_char_cnt = []
    for words in lines:
        t = 0
        for word in words:
            width = word["Geometry"]["BoundingBox"]["Width"]
            t = max(t, 1/width * len(word["Text"]))
        line_char_cnt.append(math.ceil(t))

    def get_line_from_words(words, line_idx) -> str:
        """
            获取一行的表示,
            按一行的比例,那么一行最多line_char_cnt[line_idx]个字符
            但是由于每一行预计的字符数量不一致
            1. 首先在每行内部按照每行的预计字符进行排布
            2. 将行内部排布好
            一行最多个字符
        """
        save = []
        
        left = words[0]["Geometry"]["BoundingBox"]["Left"]
        width = words[0]["Geometry"]["BoundingBox"]["Width"]
        text = words[0]["Text"]
        save.append([len(text),text])
        use_char_cnt = len(text)
        last_end = left + width
        for i in range(1, len(words)):
            word = words[i]
            text = word["Text"]
            width = word["Geometry"]["BoundingBox"]["Width"]
            left = word["Geometry"]["BoundingBox"]["Left"]
            left_space_cnt = math.ceil(
                (left - last_end) * line_char_cnt[line_idx])
            save.append([left_space_cnt, " " * left_space_cnt])
            save.append([len(text), text])
            last_end = left + width
            use_char_cnt += len(text) + left_space_cnt
        
        # 处理左端的空格和有端的空格
        left_rate = words[0]["Geometry"]["BoundingBox"]["Left"]
        right_rate = 1 - words[-1]["Geometry"]["BoundingBox"]["Left"] - words[-1]["Geometry"]["BoundingBox"]["Width"]
        max_line_char_cnt = max(line_char_cnt)
        rest_char_cnt = max_line_char_cnt - use_char_cnt
        left_space_cnt = math.ceil(left_rate / (left_rate + right_rate) * rest_char_cnt)
        right_space_cnt = rest_char_cnt - left_space_cnt
        save.insert(0, [left_space_cnt, " " * left_space_cnt])
        save.append([right_space_cnt, " " * right_space_cnt])
        # merge to one line
        ret_str = ""
        for cnt, text in save:
            if cnt == len(text):
                ret_str = ret_str + text
            else:
                raise ValueError("cnt < len(text)")
        return ret_str
    ret = []
    for i, words in enumerate(lines):
        ret.append(get_line_from_words(words,i))
    # 去除左侧的多余的空格和右侧多余的空格
    min_left_space_cnt = min([len(line) - len(line.lstrip()) for line in ret])
    min_right_space_cnt = min([len(line) - len(line.rstrip()) for line in ret])
    for i in range(len(ret)):
        ret[i] = ret[i][min_left_space_cnt:len(ret[i]) - min_right_space_cnt]
    return "\n".join(ret)


def get_word_count(all_lines: str) -> int:
    ans = 0
    all_lines = all_lines.split("\n")
    for i, line in enumerate(all_lines):
        ans += len(line.split())
    return ans


def main():
    ocr_data = open_json("./ffdw0217_p8.json")
    article = get_lines_by_words(ocr_data)
    print(article)
    with open("article.txt", "w", encoding="utf-8") as f:
        f.write(article)
    print("char total: ", get_word_count(article))


if __name__ == "__main__":
    main()
