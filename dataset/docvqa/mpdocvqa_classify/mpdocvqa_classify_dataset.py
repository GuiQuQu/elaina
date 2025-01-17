import os
import json

from dataset.base_dataset import BaseDataset, prepare_data_and_preprocessor
from utils.register import Register

def open_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["data"]


@prepare_data_and_preprocessor
@Register(name="mpdocvqa_classify_dataset")
class MPDocVQAClassifyDataset(BaseDataset):
    def __init__(self, preprocess_config, dataset_path: str, split:str="train" ) -> None:
        super().__init__(preprocess_config)

        self.dataset_path = os.path.join(dataset_path, f"{split}.json")
        self.ocr_dir = os.path.join(dataset_path, "ocr")
        self.image_dir = os.path.join(dataset_path, "images")

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
            "type": "dataset.docvqa.internvl2_preprocessor.InternVL2Preprocessor",
            "model_path": "/home/klwang/pretrain-model/InternVL2-2B",
        },
    )
    print(len(dataset))
