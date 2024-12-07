"""
    将输出的测试集的结果转换为可以上传到mpdocvqa leaderboard的格式
"""

import json


def read_input(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def read_testdata(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        data = data['data']
    qid2item = {}
    for item in data:
        qid = item["questionId"]
        qid2item[qid] = item
    return qid2item


def generate_answer_json(save_path,pred_data, testdataset):
    """
    save_path:保存的路径
    pred_data:预测的结果
    testdataset:测试集
    """
    result = []
    for item in pred_data:
        questionId = item["qid"]
        answer = item["model_output"]
        top1_page_id = item['documents'][0]['page_id']
        test_item = testdataset[questionId]
        page_idx = -1
        for idx, page_id in enumerate(test_item['page_ids']):
            if page_id == top1_page_id:
                page_idx = idx
        if page_idx == -1:
            raise ValueError(f"qid={questionId}, page_idx == -1")
        # pass
        # pred_data忘记输出page_idx了，只能通过page_id自己来找了...
        result.append({
            "questionId": questionId,
            "answer": answer,
            "answer_page": page_idx
        })
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    pred_data_path = "testdataset_result/MPDocVQA/base_qwen2vl/qwen2vl_vqa_ocr_output/checkpoint-4000-result.json"
    
    pred_data = read_input(pred_data_path)
    
    testdataset_path = "/root/autodl-tmp/MPDocVQA/test.json"
    
    qid2item = read_testdata(testdataset_path)

    save_path = pred_data_path.replace(".json", "_mpdocvqa_format.json")
    generate_answer_json(
        save_path=save_path,
        pred_data=pred_data,
        testdataset=qid2item
    )
