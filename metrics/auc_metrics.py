from typing import Tuple
import pandas as pd

from metrics.base_metrics import BaseMetrics
from utils.register import Register
from logger import logger

# 计算所有的正负样本对中，正样本排在负样本前面的概率
# 设正样本m个,负样本n个
# 所有的正负样本对数量: m*n
# 看所有的正样本
# 对于正样本概率的最大的，假设排序编号是rank1, 则比他概率小于等于的样本数是rank1个,其中正样本m个
# 对于正样本概率的第二大的，假设排序编号是rank2, 则比他概率小于等于的样本数是rank2个,其中正样本m-1个
# 以此类推
# rank1 + rank2 + ... + rankm  - (m + m-1 + ... + 1)

def calc_auc(y_true, y_pred):
    pair = list(zip(y_true, y_pred))
    pair = sorted(pair, key=lambda x: x[1])
    df = pd.DataFrame(
        [[x[0], x[1], float(i + 1)] for i, x in enumerate(pair)],
        columns=["y_true", "y_pred", "rank"],
    )

    # 将预测值相同的样本的rank取平均值
    for k, v in df.y_pred.value_counts().items():
        if v > 1:
            rank_mean = df[df.y_pred == k]["rank"].mean()
            df.loc[df.y_pred == k, "rank"] = rank_mean

    pos_df = df[df.y_true == 1]
    m = pos_df.shape[0]  # 正样本数
    n = df.shape[0] - m  # 负样本数
    if m == 0 or n == 0:
        logger.error(f"AUCMetrics: only positive samples or only negative samples, m={m}, n={n}, can't calculate auc")
        return 0.5
    auc = (pos_df["rank"].sum() - m * (m + 1) / 2) / (m * n)
    return auc
    

# from sklearn.metrics import roc_auc_score

@Register(name="auc")
class AUCMetrics(BaseMetrics):
    def __init__(self, result_path, pred_key="model_output", label_key="label"):
        super().__init__(result_path)
        self.pred_key = pred_key
        self.label_key = label_key

    def compute_metrics(self) -> Tuple[float,str]:
        y_true = [item[self.label_key] for item in self.result]
        y_pred = [item[self.pred_key] for item in self.result]
        auc = calc_auc(y_true, y_pred)
        # logger.info(f"{self.result_path}=> AUCMetrics: {auc}")
        return auc, "AUCMetrics: {:.2%}".format(auc)
