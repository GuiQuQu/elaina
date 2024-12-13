import json
import os
from collections import defaultdict
import random
from typing import List
import sys

from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.append("/root/elaina")


class MPDocVQAVqaDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
    ) -> None:
        super().__init__()

        self.dataset_path = os.path.join(dataset_path, f"{split}.json")
        self.ocr_dir = os.path.join(dataset_path, "ocr")
        self.image_dir = os.path.join(dataset_path, "images")

        self.data = self.prepare_data()

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
            ret_item = dict(
                qid=qid,
                question=question,
                documents=documents,
                answers=answers,
                true_answer_page_idx=answer_page_idx,
            )
            ret_data.append(ret_item)
        return ret_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def open_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["data"]


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def read_model_answers(model_pred_path):
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
        # _from = os.path.split(model_pred_path)[-2:]
        result[qid] = {
            "qid": qid,
            "pred_answer": pred_answer,
            "true_answers": true_answers,
            "from": model_pred_path,
        }
    return result


def make_multi_conversation_data(
    dataset: MPDocVQAVqaDataset, model_pred_paths: List[str]
):
    """
    通过规则的形式构造多轮对话的CoT数据
    这个CoT仅仅对于答案选择具有思考，对于问题分析没有思考，该多轮对话数据中没有system message
    """

    def user_prompt1(image_placeholder, question):
        prompt = """There is a document image and the question about the document image
Image: {image}
Question: {question}
You will be given some answers. 
Here are some answers provided for you.
Please judge whether these answers are correct or not."""
        return prompt.format(image=f"{image_placeholder}", question=question)

    def judge_prompt(idx, answer):
        """
        User: 'Answer x: ABC'
        """
        return f"Answer {idx}: {answer}"

    def judge_result_prompt(pred_answer: str, true_answers: List[str]):
        """
        assitant: 'True/False'
        """
        pred_answer = pred_answer.strip().lower()
        true_answers = [true_answer.strip().lower() for true_answer in true_answers]
        result = "True" if pred_answer in true_answers else "False"
        return f"{result}"

    def user_summary_instruction():
        return "Please summarize your judgment on these answers."

    def assistant_summary_result(candidate_model_answers, label_answers):
        prompt = "Summary:\n"
        # 统计正确答案和错误答案以及每个答案出现的次数
        appear_d = defaultdict(list)
        cnt_d = defaultdict(int)
        for model_pred in candidate_model_answers:
            pred_answer = model_pred["pred_answer"]
            # 不同的答案按照大小写不同可能会不同
            key = pred_answer.strip().lower()
            appear_d[key].append(pred_answer)
            cnt_d[key] += 1

        max_times = max(cnt_d.values())
        max_elems = [key for key, cnt in cnt_d.items() if cnt == max_times]

        all_keys = list(cnt_d.keys())
        true_answers = []
        false_answers = []
        for key in all_keys:
            if judge_result_prompt(key, label_answers) == "True":
                true_answers.append((appear_d[key], cnt_d[key]))
            else:
                false_answers.append((appear_d[key], cnt_d[key]))

        # 按照出现次数排序
        true_answers = sorted(true_answers, key=lambda x: x[1], reverse=True)
        false_answers = sorted(false_answers, key=lambda x: x[1], reverse=True)
        choice_true_answers = [(random.choice(a), cnt) for a, cnt in true_answers]
        choice_true_answers = sorted(
            choice_true_answers, key=lambda x: x[1], reverse=True
        )
        choice_false_answers = [(random.choice(a), cnt) for a, cnt in false_answers]
        choice_false_answers = sorted(
            choice_false_answers, key=lambda x: x[1], reverse=True
        )

        # 添加到prompt中
        prompt += "True answers:\n"
        if len(choice_true_answers) == 0:
            prompt += "None\n"
        else:
            for a, cnt in choice_true_answers:
                prompt += f"{a}, appear {cnt} times\n"

        prompt += "False answers:\n"
        if len(choice_false_answers) == 0:
            prompt += "None\n"
        else:
            for a, cnt in choice_false_answers:
                prompt += f"{a}, appear {cnt} times\n"
        prompt += "\n"

        # 根据true_answers,false_answers的情况添加分析prompt
        if len(choice_true_answers) == 0:
            prompt += (
                "There is no true answer.\nI should give the answer by myself.\n\n"
            )
        else:
            for a, cnt in choice_true_answers:
                prompt += f"{a} is the true answer\n"
                prompt += f"It appears {cnt} times\n"
                # cnt == max_times 并且没有其他元素的出现次数和它一样
                if cnt == max_times and len(max_elems) == 1:
                    prompt += f"{a} has the highest number of occurrences, which is {cnt} times.\n"
                prompt += "\n"

        final_answer = (
            choice_true_answers[0][0]
            if len(choice_true_answers) > 0
            else random.choice(label_answers)
        )
        prompt += f"My final answer is {final_answer}"
        return prompt

    candidate_model_answers = [
        read_model_answers(model_pred_path) for model_pred_path in model_pred_paths
    ]
    new_data = []
    for i, item in enumerate(tqdm(dataset)):
        qid = item["qid"]
        question = item["question"]
        documents = item["documents"]
        true_answer_page_idx = item["true_answer_page_idx"]
        true_page = documents[true_answer_page_idx]
        true_image_path = true_page["image_path"]
        true_ocr_path = true_page["ocr_path"]
        true_answers = item.get("answers", ["fake label"])

        # {'qid': '0', 'pred_answer': 'A', 'from': 'model1'}
        item_candidate_answers = [
            model_pred[qid] for model_pred in candidate_model_answers
        ]

        multi_conv = []
        prompt1 = (
            user_prompt1("<|image|>", question)
            + "\n"
            + judge_prompt(0 + 1, item_candidate_answers[0]["pred_answer"])
        )
        model_pred_idx = 1

        # first
        multi_conv.append(
            {"role": "user", "content": [{"type": "text", "text": prompt1}]}
        )
        multi_conv.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": judge_result_prompt(
                            item_candidate_answers[0]["pred_answer"], true_answers
                        ),
                    }
                ],
            }
        )

        # judge
        for a in item_candidate_answers[1:]:
            pred_answer = a["pred_answer"]
            true_answers = a["true_answers"]
            judege_answer = judge_prompt(model_pred_idx + 1, pred_answer)
            multi_conv.append(
                {"role": "user", "content": [{"type": "text", "text": judege_answer}]}
            )
            multi_conv.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": judge_result_prompt(pred_answer, true_answers),
                        }
                    ],
                }
            )
            model_pred_idx += 1

        # summary and answer
        multi_conv.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": user_summary_instruction()}],
            }
        )
        multi_conv.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": assistant_summary_result(
                            item_candidate_answers, true_answers
                        ),
                    }
                ],
            }
        )
        #
        item["cot_conversation"] = multi_conv
        item['candidate_answers'] = item_candidate_answers
        new_data.append(item)
    return new_data

def generate_in_evaldata():
    pass

def generate_in_testdata():
    pass


def main():
    split = 'test'
    # 选择一些模型的输出结果
    # model_pred_paths = [
    #     # base_internvl2 internvl2-2b w/o ocr
    #     "outputs/MPDocVQA/vqa_output_lr1e5/test_result/checkpoint-4000-result.json",
    #     # base_internvl2 qwen2vl-2b w/ ocr
    #     "outputs/MPDocVQA/qwen2vl_vqa_ocr_output/test_result/checkpoint-4000-result.json",
    #     "outputs/MPDocVQA/base_clip/internvl_lora_vqa_output/checkpoint-4000-result.json",
    # ]
    
    model_pred_paths = [
        "testdataset_result/MPDocVQA/base_internvl2/qwen2vl_vqa_ocr_output/checkpoint-4000-result.json",
        "testdataset_result/MPDocVQA/base_internvl2/internvl2_vqa_result/checkpoint-4000-result.json",
        "testdataset_result/MPDocVQA/base_clip/internvl_lora_vqa_output/checkpoint-3000-result.json"
    ]
    dataset = MPDocVQAVqaDataset(dataset_path="/root/autodl-tmp/MPDocVQA", split=split)


    new_data = make_multi_conversation_data(dataset, model_pred_paths)
    output_path = f"outputs/cot_data/{split}_rule_cot_data.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
