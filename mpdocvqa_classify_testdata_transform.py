"""
    将输出的测试集的结果转换为可以上传到mpdocvqa leaderboard的格式,
    仅仅处理分类的结果，用来查看测试的页面预测准确率
"""

import json
from collections import defaultdict

def read_input(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 排序，转dict
    qid2items = defaultdict(list)
    for item in data:
        qid = item["qid"]
        qid2items[qid].append(item)
    
    for qid in qid2items:
        qid2items[qid] = sorted(qid2items[qid], key=lambda x: x["model_output"], reverse=True)

    return qid2items


def read_testdata(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        data = data["data"]
    qid2item = {}
    for item in data:
        qid = item["questionId"]
        qid2item[qid] = item
    return qid2item


def generate_answer_json(save_path, pred_data, testdataset):
    result = []
    for qid, items in pred_data.items():
        top1_item = items[0]
        top1_page_id = top1_item["image_path"].split("/")[-1].split(".")[0]
        test_item = testdataset[qid]
        page_idx = -1
        for idx, page_id in enumerate(test_item["page_ids"]):
            if page_id == top1_page_id:
                page_idx = idx
                break
        if page_idx == -1:
            raise ValueError(f"qid={qid}, page_idx == -1")
        result.append(
            {
                "questionId": qid,
                "answer": "fake_answer for page accuracy",
                "answer_page": page_idx,
            }
        )
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    classify_data_path = "testdataset_result/MPDocVQA/ensemble/classify_2cand/checkpoint-3000-result.json"
    classify_data_path = "testdataset_result/MPDocVQA/qwen2vl_classify_output/test_result/checkpoint-41184-result.json"
    classify_data_path = "testdataset_result/MPDocVQA/internvl2_lora_classify_output/test_result/checkpoint-41185-result.json"
    classify_data_path = "/root/elaina/testdataset_result/MPDocVQA/ensemble/classify_2cand/traditional_ensemble_result.json"
    classify_data_path = "/root/elaina/testdataset_result/MPDocVQA/ensemble/classify_2cand/checkpoint-3613-result.json"
    pred_data = read_input(classify_data_path)
    testdataset_path = "/root/autodl-tmp/MPDocVQA/test.json"

    qid2item = read_testdata(testdataset_path)

    save_path = classify_data_path.replace(".json", "_mpdocvqa_format.json")
    generate_answer_json(save_path=save_path, pred_data=pred_data, testdataset=qid2item)
