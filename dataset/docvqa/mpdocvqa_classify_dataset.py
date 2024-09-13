import os
import json

from dataset.base_dataset import BaseDataset, prepare_data_and_preprocessor


def open_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["data"]


@prepare_data_and_preprocessor
class MPDocVQAClassifyDataset(BaseDataset):
    def __init__(self, dataset_path: str, preprocess_config) -> None:
        super().__init__(preprocess_config)

        self.dataset_path = os.path.join(dataset_path, "train.json")
        self.ocr_dir = os.path.join(dataset_path, "ocr")
        self.image_dir = os.path.join(dataset_path, "image")

    def prepare_data(self):
        data = open_data(self.dataset_path)
        ret_data = []
        for i, item in enumerate(data):
            qid = item["questionId"]
            question = item["question"]
            page_ids = item["page_ids"]
            answers = item["answers"]
            answer_page_idx = item["answer_page_idx"]
            for j, page_id in enumerate(page_ids):
                ret_item = dict(
                    qid=qid,
                    question=question,
                    page_id=page_id,
                    image_path=os.path.join(self.image_dir, page_id + ".jpg"),
                    ocr_path=os.path.join(self.ocr_dir, page_id + ".json"),
                    answers=answers,
                    true_answer_page_idx=answer_page_idx,
                    cls_label=1 if j == answer_page_idx else 0,
                )
                ret_data.append(ret_item)
        return ret_data



if __name__ == "__main__":
    dataset = MPDocVQAClassifyDataset(
        dataset_path="/home/klwang/data/MPDocVQA",
        preprocess_config={
            "type": ".InternVL2Preprocessor",
            "model_path": "/home/klwang/pretrain-model/InternVL2-2B",
        },
    )
    print(len(dataset))
