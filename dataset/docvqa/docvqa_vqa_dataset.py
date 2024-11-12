from collections import defaultdict
import os
import json
from typing import Any, Dict, List


from dataset.base_dataset import BaseDataset, prepare_data_and_preprocessor
from utils.register import Register

def open_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["data"]


@prepare_data_and_preprocessor
@Register(name="docvqa_vqa_dataset")
class DocVQAVqaDataset(BaseDataset):
    def __init__(
        self, preprocess_config, dataset_path: str, file_name: str = "train_v1.0_withQT"
    ) -> None:
        super().__init__(preprocess_config)
        self.dataset_path = os.path.join(dataset_path, f"{file_name}.json")
        self.ocr_dir = os.path.join(dataset_path, "ocr")
        self.image_dir = os.path.join(dataset_path, "images")
    
    def prepare_data(self) -> List[Dict[str, Any]]:
        data = open_data(self.dataset_path)
        ret_data = []
        for _, item in enumerate(data):
            qid = item['questionId']
            question = item['question']
            question_type = item['question_types']
            page_id = item['image'].split('/')[-1].split('.')[0]
            image_path = os.path.join(self.image_dir, f"{page_id}.png")
            ocr_path = os.path.join(self.ocr_dir, f"{page_id}.json")
            answers = item.get('answers', ["No Answer"])
            ret_data.append(dict(
                qid=qid,
                question=question,
                question_type=question_type,
                image_path=image_path,
                ocr_path=ocr_path,
                answers=answers
            ))
        return ret_data
