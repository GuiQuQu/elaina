from collections import defaultdict
from typing import Tuple
import json

from metrics.docvqa_metrics.anls_metrics import ANLSMetrics
from utils.register import Register

@Register(name="mp_anls")
class MPDocVQAANLSMetrics(ANLSMetrics):
    def __init__(
        self,
        result_path,
        pred_replace_star: bool = False,
        pred_replace_n: bool = False,
        pred_key: str = "model_output",
        answers_key: str = "answers",
        cls_label_key: str = "classify_label",
        cls_score_key: str = "score",
    ):
        super().__init__(
            result_path,
            pred_replace_star=pred_replace_star,
            pred_replace_n=pred_replace_n,
            pred_key=pred_key,
            answers_key=answers_key,
        )
        self.cls_label_key = cls_label_key
        self.cls_score_key = cls_score_key
        self.groupby_result()

    def groupby_result(self):
        """
        根据qid进行分组,用于后续的指标计算
        """
        qid2items = defaultdict(list)
        for _, item in enumerate(self.result):
            qid = item["qid"]
            # answers = item[self.answers_key]
            # pred = item[self.pred_key]
            # cls_label = item[self.cls_label_key]
            # cls_score = item[self.cls_score_key]
            qid2items[qid].append(item)
        result = []

        for qid, items in qid2items.items():
            items = sorted(items, key=lambda x: x[self.cls_score_key], reverse=True)
            top1_item = items[0]
            answers = top1_item[self.answers_key]
            pred = top1_item[self.pred_key]
            result.append(
                {
                    "qid": qid,
                    "top1_cls_label": top1_item[self.cls_label_key],
                    self.pred_key: pred,
                    self.answers_key: answers,
                    "documents": items,
                }
            )
        self.result = result

    def compute_metrics(self) -> Tuple[float, str]:

        all_anls = 0.0
        metrics_details = []
        for _, item in enumerate(self.result):
            answers = item[self.answers_key]
            pred = item[self.pred_key]
            single_anls = self._ls_multiple(pred, answers)
            all_anls += single_anls
            item["anls"] = single_anls
            metrics_details.append(item)
        anls = all_anls / len(self.result)
        # save metrics_details to a file
        save_path = self.result_path.replace(".json", "_anls_detials.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(metrics_details, f, ensure_ascii=False, indent=2)
        return anls, f"all_anls:{all_anls}, count:{len(self.result)}"


if __name__ == "__main__":
    metrics = MPDocVQAANLSMetrics(
        result_path="outputs/MPDocVQA/ally_output/test_result/checkpoint-40000-result.json",
        pred_key="resp",
        answers_key="answers",
    )
    print(metrics.compute_metrics())
