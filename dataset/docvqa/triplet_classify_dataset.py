from collections import defaultdict
import os
import json
import random
from typing import Any, Dict, List


from dataset.base_dataset import BaseDataset, prepare_data_and_preprocessor
from utils.register import Register


def open_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["data"]


@prepare_data_and_preprocessor
@Register(name="triplet_classify_dataset")
class TripletClassifyDataset(BaseDataset):
    """
    用户三元组损失训练
    如果当前问题对应的文档只有一个图像，
    那么会在全部数据中为其随机采样一个其他的文档加入其中
    """

    def __init__(
        self,
        preprocess_config,
        dataset_path: str,
        split: str = "train",
    ) -> None:
        super().__init__(preprocess_config)

        self.dataset_path = os.path.join(dataset_path, f"{split}.json")
        self.ocr_dir = os.path.join(dataset_path, "ocr")
        self.image_dir = os.path.join(dataset_path, "images")
        self.cache_page_id = None

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
            if len(documents) == 1:
                positive_page_id = documents[0]["page_id"]
                negative_page_id = self.sample_negative_page_id(
                    positive_page_id, new_page_idx=len(documents), data=data
                )
                documents.append(negative_page_id)

            ret_item = dict(
                qid=qid,
                question=question,
                documents=documents,
                answers=answers,
                true_answer_page_idx=answer_page_idx,
            )
            ret_data.append(ret_item)
        return ret_data

    def sample_negative_sample_idx(self, idx, data):
        """
        Sample negative samples from the idx item
        """
        length = len(data)
        idx_list = list(range(length))
        idx_list.remove(idx)
        negative_idx = random.choice(idx_list)
        return negative_idx

    def get_all_page_ids(self, data):
        cache_page_id = set()
        for item in data:
            for page_id in item["page_ids"]:
                cache_page_id.add(page_id)
        self.cache_page_id = list(cache_page_id)

    def sample_negative_page_id(self, page_id, new_page_idx, data):
        if self.cache_page_id is None:
            self.get_all_page_ids(data)

        # sample
        while True:
            negative_page_id = random.choice(self.cache_page_id)
            if negative_page_id != page_id:
                break

        negative_image_path = os.path.join(self.image_dir, negative_page_id + ".jpg")
        negative_ocr_path = os.path.join(self.ocr_dir, negative_page_id + ".json")
        ret = dict(
            page_idx=new_page_idx,
            page_id=negative_page_id,
            image_path=negative_image_path,
            ocr_path=negative_ocr_path,
            note="sample from other question",
        )
        return ret

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
