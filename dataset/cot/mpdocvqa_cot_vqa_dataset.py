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


@Register(name="mpdocvqa_vqa_cot_dataset_include_func")
def include_func(item):
    cot_json_path = item["cot_json_path"]
    if not os.path.exists(cot_json_path):
        return False
    return True


@prepare_data_and_preprocessor
@Register(name="mpdocvqa_vqa_cot_dataset")
class MPDocVQAVqaDataset(BaseDataset):
    def __init__(
        self,
        preprocess_config,
        dataset_path: str,
        cot_data_dir: str,
        split: str = "train",
        include_func=None,
        exclude_func=None,
    ) -> None:
        super().__init__(preprocess_config, include_func, exclude_func)

        self.dataset_path = os.path.join(dataset_path, f"{split}.json")
        self.ocr_dir = os.path.join(dataset_path, "ocr")
        self.image_dir = os.path.join(dataset_path, "images")
        self.cot_data_dir = cot_data_dir

    def prepare_data(self):
        data = open_data(self.dataset_path)
        ret_data = []

        for i, item in enumerate(data):
            qid = item["questionId"]
            question = item["question"]
            page_ids = item["page_ids"]
            # answers = item["answers"]
            answers = item.get("answers", ["fake label"])
            # answer_page_idx = item["answer_page_idx"]
            answer_page_idx = item.get("answer_page_idx", -1)
            if answer_page_idx == -1:
                raise ValueError(
                    f"answer_page_idx is -1 for qid: {qid}, There will be no true answer page for this question."
                )
            true_page_id = page_ids[answer_page_idx]
            cot_json_path = os.path.join(
                self.cot_data_dir, f"{qid}#{true_page_id}.json"
            )

            documents = []
            for idx, page_id in enumerate(page_ids):
                image_path = os.path.join(self.image_dir, page_id + ".jpg")
                ocr_path = os.path.join(self.ocr_dir, page_id + ".json")
                documents.append(
                    dict(
                        page_idx=idx,
                        page_id=page_id,
                        image_path=image_path,
                        ocr_path=ocr_path,
                    )
                )

            ret_item = dict(
                qid=qid,
                question=question,
                documents=documents,
                answers=answers,
                true_answer_page_idx=answer_page_idx,
                cot_json_path=cot_json_path,
            )
            ret_data.append(ret_item)
        return ret_data
