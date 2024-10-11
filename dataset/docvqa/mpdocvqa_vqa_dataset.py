from collections import defaultdict
import os
import json
from typing import Any, Dict, List


from dataset.base_dataset import BaseDataset, prepare_data_and_preprocessor


def open_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["data"]


@prepare_data_and_preprocessor
class MPDocVQAVqaDataset(BaseDataset):
    def __init__(
        self,
        preprocess_config,
        dataset_path: str,
        classify_result_path: str = None,  # 暂时不使用这部分内容，数据部分没有分类的score
        split: str = "train",
    ) -> None:
        super().__init__(preprocess_config)

        self.dataset_path = os.path.join(dataset_path, f"{split}.json")
        self.ocr_dir = os.path.join(dataset_path, "ocr")
        self.image_dir = os.path.join(dataset_path, "images")
        if classify_result_path is None:
            self.classify_result_path = os.path.join(
                dataset_path, f"classify_result_{split}.json"
            )
        else:
            self.classify_result_path = classify_result_path

    def prepare_data(self):
        data = open_data(self.dataset_path)
        ret_data = []

        for i, item in enumerate(data):
            qid = item["questionId"]
            question = item["question"]
            page_ids = item["page_ids"]
            answers = item["answers"]
            answer_page_idx = item["answer_page_idx"]
            
            documents = []
            for page_id in page_ids:
                image_path = os.path.join(self.image_dir, page_id + ".jpg")
                ocr_path = os.path.join(self.ocr_dir, page_id + ".json")
                documents.append(
                    dict(
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
            )
            ret_data.append(ret_item)
        return ret_data


    def groupby_classify_result(self) -> Dict[str, List[Dict[str, Any]]]:

        ret_dict = defaultdict(dict)
        with open(self.classify_result_path, "r", encoding="utf-8") as f:
            classify_result = json.load(f)

        for i, item in enumerate(classify_result):
            qid = item["qid"]
            page_id = item["image_path"].split("/")[-1].split(".")[0]
            score = item["model_output"]
            ret_dict[qid][page_id] = score 
        return ret_dict
