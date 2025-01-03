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


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@Register(name="true_answer_in_any_candidate_answers")
def include_func(item):
    true_answers = item["answers"]
    true_answers = [a.lower().strip() for a in true_answers]
    candidate_answers = item["candidate_answers"]
    for ca in candidate_answers:
        if ca["pred_answer"].lower().strip() in true_answers:
            return True
    return False


def read_model_answers(model_pred_path, confidence: int = 1):
    """
    读取模型预测的答案
    """
    agent_result = load_json(model_pred_path)
    # 整理为dict形式
    result = defaultdict(dict)
    for item in agent_result:
        qid = item["qid"]
        pred_answer: str = item["model_output"]
        true_answers: List[str] = item["answers"]
        result[qid] = {
            "qid": qid,
            "pred_answer": pred_answer,
            "true_answers": true_answers,
            "from": model_pred_path,
            "confidence": confidence,
        }
    return result


@prepare_data_and_preprocessor
@Register(name="ensemble_mpdocvqa_vqa_dataset")
class EnsembleMPDocVQAVqaDataset(BaseDataset):
    def __init__(
        self,
        preprocess_config,
        dataset_path: str,
        ensemble_result_paths=[],
        split: str = "train",
        include_func="true_answer_in_any_candidate_answers",
        exclude_func=None,
    ) -> None:
        super().__init__(preprocess_config, include_func, exclude_func)

        self.dataset_path = os.path.join(dataset_path, f"{split}.json")
        self.ocr_dir = os.path.join(dataset_path, "ocr")
        self.image_dir = os.path.join(dataset_path, "images")
        self.ensemble_result_paths = ensemble_result_paths
        # 读取其他模型给出的候选答案
        if (
            len(ensemble_result_paths) > 0
            and isinstance(ensemble_result_paths[0], list)
            and len(ensemble_result_paths[0]) == 2
        ):
            self.candidate_answers_list = [
                read_model_answers(path, confidence)
                for path, confidence in ensemble_result_paths
            ]
        else:
            self.candidate_answers_list = [
                read_model_answers(path) for path in ensemble_result_paths
            ]

    def prepare_data(self):
        data = open_data(self.dataset_path)
        ret_data = []

        for i, item in enumerate(data):
            qid = item["questionId"]
            question = item["question"]
            page_ids = item["page_ids"]
            
            # answers = item["answers"]
            answers = item.get("answers", ['fake_answer'])
            # answer_page_idx = item["answer_page_idx"]
            answer_page_idx = item.get("answer_page_idx", -1)
            
            true_page_id = page_ids[answer_page_idx]

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
            candidate_answers = [
                ca_dict[qid] for ca_dict in self.candidate_answers_list
            ]
            ret_item = dict(
                qid=qid,
                question=question,
                documents=documents,
                answers=answers,
                true_answer_page_idx=answer_page_idx,
                candidate_answers=candidate_answers,
            )
            ret_data.append(ret_item)
        return ret_data
