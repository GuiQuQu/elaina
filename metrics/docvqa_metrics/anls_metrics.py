from typing import List, Tuple
import Levenshtein
import json

from metrics.base_metrics import BaseMetrics
from utils.register import Register
from logger import logger

@Register(name="anls")
class ANLSMetrics(BaseMetrics):
    def __init__(
        self, result_path, pred_replace_star: bool = False, pred_replace_n: bool = False,
        pred_key:str = "model_output", answers_key:str = "answers"
    ):
        super().__init__(result_path)
        self.pred_replace_star = pred_replace_star
        self.pred_replace_n = pred_replace_n
        self.pred_key = pred_key
        self.answers_key = answers_key

    def compute_metrics(self) -> Tuple[float, str]:
        all_anls = 0.0
        metrics_details = []
        for _, item in enumerate(self.result):
            answers = item[self.answers_key]
            pred = item[self.pred_key]
            single_anls = self._ls_multiple(pred, answers)
            all_anls += single_anls
            item['anls'] = single_anls
            metrics_details.append(item)
        anls = all_anls / len(self.result)
        # save metrics_details to a file
        save_path = self.result_path.replace(".json", "_anls_detials.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_details, f, ensure_ascii=False, indent=2)
        return anls, f"all_anls:{all_anls}, count:{len(self.result)}"
        
    def _ls(self, s1: str, s2: str, threshold=0.5):
        s1 = s1.lower().strip()
        s2 = s2.lower().strip()
        nls = Levenshtein.distance(s1, s2) / max(len(s1), len(s2))
        return 1 - nls if nls < threshold else 0.0

    def _ls_multiple(self, pred: str, answers: List[str], threshold=0.5):
        if self.pred_replace_star:
            pred = pred.replace("*", "")
        if self.pred_replace_n:
            pred = pred.replace("\n", "")
        return max([self._ls(pred, ans, threshold) for ans in answers])
