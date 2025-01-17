import os
import json
from PIL import Image
import torch
from open_clip.transform import image_transform_v2, PreprocessCfg
from open_clip import get_tokenizer


from dataset.base_preprocessor import BasePreprocessor
from logger import logger
from utils.register import Register

HF_HUB_PREFIX = "hf-hub:"


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@Register(name="mpdocvqa_classify_triplet_preprocessor")
class MPDocVQAClassifyTripletPreprocessor(BasePreprocessor):
    def __init__(self, model_path, max_negative_length:int = 15) -> None:
        super().__init__()
        config_path = os.path.join(model_path, "open_clip_config.json")
        self.config = load_json(config_path)
        pp_config = PreprocessCfg(**self.config["preprocess_cfg"])
        self.train_transform = image_transform_v2(pp_config, is_train=True)
        self.tokenizer = get_tokenizer(HF_HUB_PREFIX + model_path)
        self.max_negative_length = max_negative_length

    def preprocess(self, item):
        qid = item["qid"]
        question = item["question"]
        documents = item["documents"]
        true_answer_page_idx = item["true_answer_page_idx"]
        true_page = documents[true_answer_page_idx]
        true_image_path = true_page["image_path"]
        true_ocr_path = true_page["ocr_path"]
        answers = item["answers"]

        wrong_image_path = [
            page["image_path"]
            for page in documents
            if page["image_path"] != true_image_path
        ]

        # anchor
        input_ids = self.tokenizer([question]).squeeze(0)  # [L]

        # positive
        positive_image = Image.open(true_image_path).convert("RGB")
        negative_image_list = [
            Image.open(image_path).convert("RGB") for image_path in wrong_image_path
        ]
        negative_image_list = negative_image_list[:self.max_negative_length]
        extra = dict(
            qid=qid,
            true_image_path=true_image_path,
            true_ocr_path=true_ocr_path,
            answers=answers,
            documents=documents,
        )
        model_inputs = dict(
            extra=extra,
            anchor_input_ids=input_ids,
            positive_image=self.train_transform(positive_image).squeeze(0),  # [C,H,W]
            negative_images=torch.stack([
                self.train_transform(image).squeeze(0) for image in negative_image_list
            ]),
        )

        return model_inputs

# 用clip的test preprocessor完全可以    
# @Register(name="mpdocvqa_classify_triplet_test_preprocessor")
# class MPDocVQAClassifyTripletPreprocessor(BasePreprocessor):
#     def __init__(self, model_path) -> None:
#         super().__init__()
#         config_path = os.path.join(model_path, "open_clip_config.json")
#         self.config = load_json(config_path)
#         pp_config = PreprocessCfg(**self.config["preprocess_cfg"])
#         self.train_transform = image_transform_v2(pp_config, is_train=True)
#         self.tokenizer = get_tokenizer(HF_HUB_PREFIX + model_path)
