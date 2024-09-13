import os
from tqdm import tqdm
from sp_ocr import sp_layout_star_from_json_path

def generate_docvqa_layout(ocr_dir, layout_dir):
    """
    generate docvqa layout from ocr data,save to layout_dir
    """
    if not os.path.exists(layout_dir):
        os.makedirs(layout_dir)
    for file in tqdm(os.listdir(ocr_dir)):
        if file.endswith(".json"):
            json_path = os.path.join(ocr_dir, file)
            layout = sp_layout_star_from_json_path(json_path)
            layout_path = os.path.join(layout_dir, file.replace(".json", ".txt"))
            with open(layout_path, "w", encoding="utf-8") as f:
                f.write(layout)

if __name__ == "__main__":
    ocr_dir = "/home/klwang/data/spdocvqa-dataset/ocr"
    layout_dir = "/home/klwang/data/spdocvqa-dataset/layout"
    generate_docvqa_layout(ocr_dir, layout_dir)