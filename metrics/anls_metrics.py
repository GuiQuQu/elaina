"""
    计算anls指标
"""

from typing import List, Tuple
import Levenshtein
import os
import json


class MPDocVQAItem(object):
    def __init__(
        self,
        qid,
        question,
        # predict
        predictions: List[Tuple[float, str]],  # predict result for every page
        # support message
        image_paths: List[str] = None,
        ocr_paths: List[str] = None,
        layout_paths: List[str] = None,
        exec_time: float = None,
        # label
        answers: List[str] = None,
        answer_page_idx: int = None,
    ):
        self.qid = qid
        self.question = question
        self.predictions = predictions
        self.answers = answers
        self.answer_page_idx = answer_page_idx
        self.image_paths = [str(image_path) for image_path in image_paths]
        self.ocr_paths = [str(ocr_path) for ocr_path in ocr_paths]
        self.layout_paths = (
            [str(layout_path) for layout_path in layout_paths]
            if layout_paths[0] is not None
            else layout_paths
        )
        self.eval_mode = answers is not None and answer_page_idx is not None

        # pred_dict

    def __repr__(self) -> str:
        return json.dumps(self.to_result_dict(),ensure_ascii=False)
    
    @property
    def true_page_idx(self):
        return self.get_true_label_dict()["true_index"]
    
    @property
    def pred_page_idx(self):
        return self.get_pred_dict()["pred_index"]

    @property
    def predict_answer(self) -> str:
        return self.get_pred_dict()["pred_answer"]

    @property
    def true_answers(self) -> List[str]:
        if self.eval_mode:
            return self.get_true_label_dict()["true_answers"]
        else:
            return None

    def get_top1_predict_result(self):
        if hasattr(self, "sorted_predictions"):
            return self.sorted_predictions[0]
        sorted_predictions = sorted(
            # score, answer, idx
            [(p[0], p[1], i) for i, p in enumerate(self.predictions)],
        )
        self.sorted_predictions = sorted_predictions
        return sorted_predictions[0]

    def to_result_dict(self):
        if hasattr(self, "result_dict"):
            return self.result_dict
        result = {
            "questionId": self.qid,
            "question": self.question,
            "pred_answer": self.get_pred_dict()["pred_answer"],
            "pred_answer_idx": self.get_pred_dict()["pred_index"],
        }
        if self.eval_mode:
            result.pop("pred_answer")
            result.pop("pred_answer_idx")
            result["pred"] = self.get_pred_dict()
            result["ground_truth"] = self.get_true_label_dict()
        self.result_dict = result
        return result

    def get_pred_dict(self):
        if hasattr(self, "pred_dict"):
            return self.pred_dict
        cur_pred = self.get_top1_predict_result()
        pred_index = cur_pred[2]
        pred_dict = {
            "pred_index": cur_pred[2],
            "score": cur_pred[0],
            "pred_answer": cur_pred[1],
            "pred_image_path": self.image_paths[pred_index],
            "pred_ocr_path": self.ocr_paths[pred_index],
            "pred_layout_path": self.layout_paths[pred_index],
        }
        self.pred_dict = pred_dict
        return self.pred_dict

    def get_true_label_dict(self):
        if hasattr(self, "ground_truth"):
            return self.ground_truth
        true_index = self.answer_page_idx
        ground_truth = {
            "true_index": true_index,
            "true_answers": self.answers,
            "true_image_path": self.image_paths[true_index],
            "true_ocr_path": self.ocr_paths[true_index],
            "true_layout_path": self.layout_paths[true_index],
        }
        self.ground_truth = ground_truth
        return ground_truth


# 以前用的方法，现在弃用
def anls(
    predict_answer: List[str], ground_truth: List[List[str]], threshold=0.5
) -> float:
    """
    n = len(predict_answer),问题的数量
    predict_answer: List[str], 每个问题的预测答案
    ground_truth: List[List[str]], 每个问题的真实答案[一个问题可能存在多个]

    reference:
    https://github.com/shunk031/ANLS
    """
    res = 0.0
    n = len(predict_answer)
    for pa, gts in zip(predict_answer, ground_truth):
        y_pred = " ".join(pa.strip().lower().split())
        anls_scores: List[float] = []
        for gt in gts:
            y_true = " ".join(gt.strip().lower().split())
            anls_score = similarity(y_true, y_pred, threshold=threshold)
            anls_scores.append(anls_score)
        res += max(anls_scores)
    return res / n


# Normalized Levenshtein distance
def similarity(answer_ij: str, predict_i: str, threshold: float = 0.5) -> float:
    maxlen = max(len(answer_ij), len(predict_i))
    edit_dist = Levenshtein.distance(answer_ij, predict_i)
    nl_score = 0.0
    if maxlen != 0:
        nl_score = float(edit_dist) / float(maxlen)

    return 1 - nl_score if nl_score < threshold else 0.0


# borrow from 'https://github.com/WenjinW/LATIN-Prompt/blob/main/metric/anls.py'
class ANLS(object):
    def __init__(
        self,
        result_dir,
        experiment_name,
        dataset_name,
        replace_star: bool = True,
        replace_n: bool = True,
    ) -> None:
        super().__init__()
        self.result_dir = result_dir
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        self.experiment_name = experiment_name
        self.dataset_name = dataset_name
        self.replace_star = replace_star
        self.replace_n = replace_n

    def _ls(self, s1: str, s2: str, threshold=0.5):
        # s1 = " ".join(s1.strip().lower().split())
        # s2 = " ".join(s2.strip().lower().split())
        s1 = s1.lower().strip()
        s2 = s2.lower().strip()
        nls = Levenshtein.distance(s1, s2) / max(len(s1), len(s2))
        return 1 - nls if nls < threshold else 0.0

    def _ls_multiple(self, pred:str, answers: List[str], threshold=0.5):
        if self.replace_star:
            pred = pred.replace("*", "")
        if self.replace_n:
            pred = pred.replace("\n", "")
        return max([self._ls(pred, ans, threshold) for ans in answers])

    def compute_and_save_docvqa(
        self,
        qids: List[int],
        questions: List[str],
        predictions: List[str],
        image_paths: List[str] = None,
        ocr_paths: List[str] = None,
        layout_paths: List[str] = None,
        answers: List[List[str]] = None,
        split="val",
    ):
        """
        保存计算结果,如果answers不为None,则计算anls
        """
        if answers is not None:
            assert image_paths is not None, "for dev data, image_paths is None"
            assert ocr_paths is not None, "for dev data, ocr_paths is None"
            assert layout_paths is not None, "for dev data,layout_paths is None"
        all_anls = 0.0
        results = []
        for i in range(len(qids)):
            result = {"questionId": qids[i], "answer": predictions[i]}
            if answers is not None:
                anls = self._ls_multiple(predictions[i], answers[i])
                all_anls += anls
                result["question"] = questions[i]
                result["image_path"] = image_paths[i]
                result["ocr_path"] = ocr_paths[i]
                result["layout_path"] = layout_paths[i]
                result["ground_truth"] = answers[i]
                result.pop("answer")
                result["answer"] = predictions[i]
                result["anls"] = anls
            results.append(result)
        save_path = os.path.join(
            self.result_dir, f"{self.experiment_name}_{self.dataset_name}_{split}.json"
        )
        if answers is not None:
            score = all_anls / len(qids)
            results.insert(0, {"anls": score})
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        return all_anls / len(qids)

    def compute_and_save_mpdocvqa(
        self,
        qids: List[int],
        questions: List[str],
        predictions: List[List[Tuple[float, str]]],
        # [[(0.56,'abcxx'),...]...]
        image_paths: List[List[str]] = None,
        ocr_paths: List[List[str]] = None,
        layout_paths: List[List[str]] = None,
        true_answers: List[Tuple[int, List[str]]] = None,  # true labels
        # [(12,['abc','ABC','Abc']),...]
        split: str = "val",
    ):
        raise ValueError("use compute_and_save_mpdocvqav2() instead")
        "保存计算结果，如果answers不为None，则计算anls"
        eval_mode = true_answers is not None
        if true_answers is not None:
            assert image_paths is not None, "for dev data, image_paths is None"
            assert ocr_paths is not None, "for dev data, ocr_paths is None"
            assert layout_paths is not None, "for dev data,layout_paths is None"
        all_anls = 0.0
        results = []
        for i in range(len(qids)):
            ground_truth = None
            if eval_mode:
                true_index = true_answers[i][0]
                ground_truth = {
                    "true_index": true_index,
                    "true_answer": true_answers[i][1],
                    "true_image_path": image_paths[i][true_index],
                    "true_ocr_path": ocr_paths[i][true_index],
                    "true_layout_path": layout_paths[i][true_index],
                }
            cur_pred = sorted(
                # score, answer, idx
                [(p[0], p[1], i) for i, p in enumerate(predictions[i])],
                key=lambda x: x[0],
                reverse=True,
            )
            pred_index = cur_pred[0][2]
            pred = {
                "pred_index": pred_index,
                "score": cur_pred[0][0],
                "pred_answer": cur_pred[0][1],
                "pred_image_path": image_paths[i][pred_index],
                "pred_ocr_path": ocr_paths[i][pred_index],
                "pred_layout_path": layout_paths[i][pred_index],
            }
            # get answer and idx
            # test result
            result = {
                "questionId": qids[i],
                "answer": pred["pred_answer"],
                "pred_index": pred["pred_index"],
            }
            if eval_mode:
                anls = self._ls_multiple(
                    pred["pred_answer"], ground_truth["true_answer"]
                )
                all_anls += anls
                result.pop("answer")
                result.pop("pred_index")
                result["question"] = questions[i]
                result["anls"] = anls
                result["ground_truth"] = ground_truth
                result["pred"] = pred
            results.append(result)
        save_path = os.path.join(
            self.result_dir, f"{self.experiment_name}_{self.dataset_name}_{split}.json"
        )
        if eval_mode:
            avg_anls = all_anls / len(qids)
            results.insert(0, {"anls": avg_anls})
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        return all_anls / len(qids)

    def compute_and_save_mpdocvqav2(
        self, mpdocvqa_items: List[MPDocVQAItem], split="val"
    ):
        """
        简洁化写法
        """
        eval_mode = mpdocvqa_items[0].eval_mode
        results = []
        all_anls = 0.0
        page_pred_right_cnt = 0
        for item in mpdocvqa_items:
            result = item.to_result_dict()
            anls = self._ls_multiple(item.predict_answer, item.true_answers)
            result["anls"] = anls
            if item.pred_page_idx == item.true_page_idx:
                page_pred_right_cnt += 1
            all_anls += anls
            results.append(result)
        save_path = os.path.join(
            self.result_dir, f"{self.experiment_name}_{self.dataset_name}_{split}.json"
        )
        avg_anls = all_anls / len(mpdocvqa_items)
        if eval_mode:
            precision = page_pred_right_cnt / len(mpdocvqa_items)
            results.insert(0, {"anls": f"{avg_anls:.4f}", 'precision':f"{precision:.4f}"})
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        return avg_anls


def old_anls2new_anls(
    result_jsonl: str, data_json: str, new_result_dir, experiment_name
):
    """
    把旧的结果文件转成新的结果文件格式
    """
    import dataset.docvqa.docvqa_utils as docvqa_utils


    line = ""
    results = []
    data = docvqa_utils.load_data(data_json)
    with open(result_jsonl, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break
            result = json.loads(line)
            results.append(result)

    anls = ANLS(
        result_dir=new_result_dir,
        experiment_name=experiment_name,
        dataset_name="spdocvqa",
    )
    qids = []
    questions = []
    predictions = []
    image_paths = []
    ocr_paths = []
    layout_paths = []
    answers = []
    for result, item in zip(results, data):
        assert result["question"] == item["question"]
        qids.append(item["questionId"])
        questions.append(item["question"])
        predictions.append(result["response"])
        image_path = result["image_path"] if "image_path" in result else result["image"]
        ocr_path: str = result["ocr_path"] if "ocr_path" in result else result["ocr"]
        image_paths.append(image_path)
        ocr_paths.append(ocr_path)
        layout_path = ocr_path.replace("ocr", "layout").replace(".json", ".txt")
        layout_paths.append(layout_path)
        answers.append(item["answers"])

    anls.compute_and_save_docvqa(
        qids=qids,
        questions=questions,
        predictions=predictions,
        image_paths=image_paths,
        ocr_paths=ocr_paths,
        layout_paths=layout_paths,
        answers=answers,
        split="val",
    )
    """
    result = {
        "questionId": qids[i],
        "answer": predictions[i]
        "question": questions[i]
        'image_path': image_path[i]
        'ocr_path': ocr_path[i]
        'layout_path': layout_path[i]
        "ground_truth": answers[i]
        "anls": anls
    }
    """


if __name__ == "__main__":
    project_dir = "/home/klwang/code/GuiQuQu-docvqa-vllm-inference"
    result_jsonl = os.path.join(
        project_dir,
        "result/qwen-vl_only-image_qlora_old/qwen-vl-int4_sft_only-image_checkpoint-final.jsonl",
    )
    data_json = "/home/klwang/data/spdocvqa-dataset/val_v1.0_withQT.json"
    new_result_dir = os.path.join(project_dir, "result/qwen-vl_only-image_qlora/")
    old_anls2new_anls(
        result_jsonl=result_jsonl,
        data_json=data_json,
        new_result_dir=new_result_dir,
        experiment_name="qwen-vl-int4_sft_only-image_checkpoint-final",
    )
