import os
import json
from collections import defaultdict
from typing import List
from dataset.base_dataset import BaseDataset, prepare_data_and_preprocessor
from utils.register import Register


def open_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["data"]


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def read_classify_resuult(model_result_paths):
    result = defaultdict(list)
    for path in model_result_paths:
        model_result = load_json(path)
        for item in model_result:
            qid = item["qid"]
            page_id = item["image_path"].split("/")[-1].split(".")[0]
            model_output = item["model_output"]
            key = f"{qid}#{page_id}"
            # info
            info = dict(
                model_output=model_output,
                from_result=path
            )
            result[key].append(info)
    return result


@prepare_data_and_preprocessor
@Register(name="mpdocvqa_classify_ensemble_dataset")
class MPDocVQAClassifyDataset(BaseDataset):
    def __init__(
        self,
        preprocess_config,
        dataset_path: str,
        ensemble_result_paths: List[str],
        split: str = "train",
    ) -> None:
        super().__init__(preprocess_config)

        self.dataset_path = os.path.join(dataset_path, f"{split}.json")
        self.ocr_dir = os.path.join(dataset_path, "ocr")
        self.image_dir = os.path.join(dataset_path, "images")
        self.ensemble_result_paths = ensemble_result_paths

    def prepare_data(self):
        data = open_data(self.dataset_path)
        ret_data = []
        classify_result = read_classify_resuult(self.ensemble_result_paths)
        for i, item in enumerate(data):
            qid = item["questionId"]
            question = item["question"]
            page_ids = item["page_ids"]
            # answers = item["answers"]
            answers = item.get("answers", ["fake label"])
            # answer_page_idx = item["answer_page_idx"]
            answer_page_idx = item.get("answer_page_idx", -1)
            for j, page_id in enumerate(page_ids):
                ret_item = dict(
                    qid=qid,
                    question=question,
                    page_id=page_id,
                    image_path=os.path.join(self.image_dir, page_id + ".jpg"),
                    ocr_path=os.path.join(self.ocr_dir, page_id + ".json"),
                    answers=answers,
                    candidate_result=classify_result[f"{qid}#{page_id}"],
                    true_answer_page_idx=answer_page_idx,
                    cls_label=1 if j == answer_page_idx else 0,
                )
                ret_data.append(ret_item)
        return ret_data
