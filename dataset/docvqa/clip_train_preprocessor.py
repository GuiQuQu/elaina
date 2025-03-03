import os
from PIL import Image
import json

from dataset.base_preprocessor import BasePreprocessor
from logger import logger
from utils.register import Register

from open_clip.transform import image_transform_v2,PreprocessCfg
from open_clip import get_tokenizer

HF_HUB_PREFIX = "hf-hub:"

def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

@Register(name="clip_train_preprocessor")
class CLIPTrainPreprocessor(BasePreprocessor):
    """
        用于vqa数据集，要求一次将该问题和所有的图像都拿到
    """
    def __init__(self,model_path):
        super().__init__()
        config_path = os.path.join(model_path, "open_clip_config.json")
        self.config = load_json(config_path)
        pp_config = PreprocessCfg(**self.config['preprocess_cfg'])
        self.train_transform = image_transform_v2(pp_config,is_train = True)
        self.tokenizer = get_tokenizer(HF_HUB_PREFIX + model_path)
    

    def preprocess(self, item):
        qid = item["qid"]
        question = item["question"]
        documents = item["documents"]
        true_answer_page_idx = item["true_answer_page_idx"]
        true_page = documents[true_answer_page_idx]
        true_image_path = true_page["image_path"]
        true_ocr_path = true_page["ocr_path"]
        answers = item["answers"]

        extra = dict(
            qid=qid,
            true_image_path=true_image_path,
            true_ocr_path=true_ocr_path,
            answers=answers,
            documents=documents,
        )
        image = Image.open(true_image_path).convert("RGB")
        image = self.train_transform(image).squeeze(0) # [C,H,W]
        input_ids = self.tokenizer([question]).squeeze(0) # [L]
        model_inputs = dict(
            extra = extra,
            image=image,
            text=input_ids,
        )

        return model_inputs


@Register(name="clip_test_preprocessor")
class CLIPTrainPreprocessor(BasePreprocessor):
    """
        用于分类数据集，要求只获取一个问题和一个图像，最后输出一个相似度得分
    """
    def __init__(self,model_path):
        super().__init__()
        config_path = os.path.join(model_path, "open_clip_config.json")
        self.config = load_json(config_path)
        pp_config = PreprocessCfg(**self.config['preprocess_cfg'])
        self.test_transform = image_transform_v2(pp_config,is_train = False)
        self.tokenizer = get_tokenizer(HF_HUB_PREFIX + model_path)
    

    def preprocess(self, item):
        
        qid = item["qid"]
        question = item["question"]
        image_path = item["image_path"]
        ocr_path = item["ocr_path"]
        cls_label = item.get("cls_label", 0)
        # label = "B" if cls_label == 0 else "A"

        extra = dict(
            qid=qid,
            image_path=image_path,
            ocr_path=ocr_path,
            question=question,
            label=cls_label,
        )
        image = Image.open(image_path).convert("RGB")
        image = self.test_transform(image).squeeze(0) # [C,H,W]
        input_ids = self.tokenizer([question]).squeeze(0) # [L]
        model_inputs = dict(
            extra = extra,
            image=image,
            text=input_ids,
        )

        return model_inputs

