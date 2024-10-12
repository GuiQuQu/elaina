
from collections import defaultdict
from typing import Tuple
from metrics.base_metrics import BaseMetrics

from logger import logger

class MPDocVQPageAccuracyMetrics(BaseMetrics):
    def __init__(self, result_path, pred_key="model_output", label_key="label"):
        super().__init__(result_path)
        self.pred_key = pred_key
        self.label_key = label_key

    def compute_metrics(self) -> Tuple[float,str]:
        
        # 按照qid做groupby
        qid2items = defaultdict(list)
        for item in self.result:
            qid = item["qid"]
            qid2items[qid].append(item)

        # 计算每个qid的预测结果,取top1结果,top1结果的label为true,则该条数据预测正确
        total_count = 0
        correct_count = 0
        for qid, items in qid2items.items():
            items = sorted(items, key=lambda x: x[self.pred_key], reverse=True)
            top1_item = items[0]
            top1_label = top1_item[self.label_key]
            if top1_label:
                correct_count += 1
            total_count += 1
        output_str = f"{correct_count / total_count :.2%}[{correct_count}|{total_count}]"
        return correct_count / total_count, output_str
