"""
    目前使用的数据是qwen-vl-max生产的推理过程，只删除了最后的结论答案，并替换成了json格式的答案
"""

from collections import defaultdict
import json
import os
import random
from typing import List
from PIL import Image
import torch

from transformers import Qwen2VLProcessor
from dataset.qwen2vl.template import Qwen2VLTemplate
from dataset.base_preprocessor import BasePreprocessor
from dataset.qwen2vl.preprocess import generate_labels
from logger import logger
from utils.register import Register


prompt_template = """You are given an image and a question. 
Image: {image}
Question: {question}
Please answer the question based on the image.
You should extract the answer from the text in the image without changing the order and form of the words.
Answer: """

### prompt个格式用错了，没有做有关要求他输出cot和json格式答案的输出
prompt_template_cot = """
You are given an image and a question.
Image: {image}
Question: {question}
Please answer the question based on the image.
You should extract the answer from the text in the image without changing the order and form of the words.
Please output your think steps as detailed as possible.
Please output your answer in the following json format:
```json
{{'answer': xxx}}
```
"""

json_answer = """
Json Answer:
```json
{{"answer": "{answer}"}}
```
"""


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@Register(name="mpdocvqa_cot_vqa_qwen2vl_exclude_func")
def exclude_func(item):
    qid = item["qid"]
    if qid in [234, 237]:
        logger.info(f"qid {qid} is excluded")
        return True
    return False


@Register(name="mpdocvqa_cot_vqa_qwen2vl_test_preprocessor")
class MPDocVQACoTVQAQwen2VLTestPreprocessor(BasePreprocessor):
    def __init__(
        self,
        model_path,
        classify_result_path: str,
        max_seq_length=1024,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        system_message="You are a helpful assistant.",
    ) -> None:
        super().__init__()

        self.template = Qwen2VLTemplate()
        self.max_seq_length = max_seq_length
        self.system_message = system_message
        self.qid2classifyitems = self.groupby_classify_result(classify_result_path)
        # self.tokenizer: Qwen2Tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        # self.tokenizer.model_max_length = max_seq_length
        self.processor = Qwen2VLProcessor.from_pretrained(
            model_path, min_pixels=min_pixels, max_pixels=max_pixels
        )
        self.reverse = True
        logger.info(
            f"min_tokens {min_pixels // 28 // 28}, max_tokens {max_pixels // 28 // 28}"
        )

    def groupby_classify_result(self, classify_result_path):
        qid2items = defaultdict(dict)
        with open(classify_result_path, "r", encoding="utf-8") as f:
            classify_result = json.load(f)
        for item in classify_result:
            qid = item["qid"]
            page_id = item["image_path"].split("/")[-1].split(".")[0]
            qid2items[qid][page_id] = item["model_output"]
        return qid2items

    def get_prompt(self, answer, **kwargs):
        prompt = prompt_template_cot.format(**kwargs)
        # 按照openai的格式组织输入
        train_ret = [
            {
                "role": "user",
                "content": prompt,
            },
            {"role": "assistant", "content": answer},
        ]
        test_ret = [{"role": "user", "content": prompt}]
        return train_ret, test_ret

    def get_text(self, messages, add_generation_prompt=False):
        """
        将对话转为添加了特殊标记的文本
        """
        self.template.clear_messages()
        for message in messages:
            role = message["role"]
            msg = message["content"]
            self.template.add_message(role, msg)
        return self.template.get_prompt(add_generation_prompt)

    def preprocess(self, item):

        qid = item["qid"]
        question = item["question"]
        documents = item["documents"]
        true_answer_page_idx = item["true_answer_page_idx"]
        true_page = documents[true_answer_page_idx]
        # true_image_path = true_page["image_path"]
        # true_ocr_path = true_page["ocr_path"]
        answers = item["answers"]

        classify_items: dict = self.qid2classifyitems[qid]
        # 根据分类结果选择得分最高的score对应的文档
        for i, doc in enumerate(documents):
            page_id = doc["page_id"]
            doc["is_true_page"] = i == true_answer_page_idx
            if page_id in classify_items:
                doc["score"] = classify_items[page_id]
            else:
                raise ValueError(f"page_id: {page_id} not found in classify_result")
        documents = sorted(documents, key=lambda x: x["score"], reverse=self.reverse)
        top1_image_path = documents[0]["image_path"]
        top1_ocr_path = documents[0]["ocr_path"]

        answer = random.choice(answers).lower().strip()

        image = Image.open(top1_image_path).convert("RGB")
        _, test_conversation = self.get_prompt(
            answer=random.choice(answers),
            image=self.template.image_placeholder,
            question=question,
        )
        # train_text = self.get_text(train_conversation, add_generation_prompt=False)
        test_text = self.get_text(test_conversation, add_generation_prompt=True)
        test_inputs = self.processor(
            text=[test_text],
            images=[image],
            videos=None,
            padding="max_length",
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        model_inputs = dict()
        extra = dict(
            qid=qid,
            answers=answers,
            documents=documents,
            test_conversation=test_conversation,
        )
        model_inputs.update(
            dict(
                extra=extra,
                test_pixel_values=test_inputs["pixel_values"],
                test_image_grid_thw=test_inputs["image_grid_thw"].squeeze(),
                test_input_ids=test_inputs["input_ids"].squeeze(),
                test_attention_mask=test_inputs["attention_mask"].squeeze(),
            )
        )
        return model_inputs
