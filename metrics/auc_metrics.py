import pandas as pd

from metrics.base_metrics import BaseMetrics
from logger import logger

def calc_auc(y_true, y_pred):
    pair = list(zip(y_true, y_pred))
    pair = sorted(pair, key=lambda x: x[1])
    df = pd.DataFrame([[x[0],x[1],i+1]for i, x in enumerate(pair)],columns=['y_true', 'y_pred'])

    # 将预测值相同的样本的rank取平均值
    for k,v in df.y_pred.value_counts().items():
        if v > 1:
            rank_mean = df[df.y_pred == k]["rank"].mean()
            df.loc[df.y_pred == k, "rank"] = rank_mean

    pos_df = df[df.y_true == 1]
    m = pos_df.shape[0] # 正样本数
    n = df.shape[0] - m # 负样本数
    return (pos_df["rank"].sum() - m * (m + 1) / 2) / (m * n)

class AUCMetrics(BaseMetrics):
    def __init__(self, result_path, pred_key='model_output', label_key='label'):
        super().__init__(result_path)
        self.pred_key = pred_key
        self.label_key = label_key

    def compute_metrics(self) -> float:
        y_true = [item[self.label_key] for item in self.result]
        y_pred = [item[self.pred_key] for item in self.result]
        auc = calc_auc(y_true, y_pred)
        # logger.info(f"{self.result_path}=> AUCMetrics: {auc}")
        return auc