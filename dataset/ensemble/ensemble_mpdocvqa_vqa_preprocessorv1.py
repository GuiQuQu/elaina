"""
    vqa任务,prompt构造均要求模型从所有的候选答案中选择一个输出
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
from dataset.qwen2vl.preprocess import generate_labels, check_over_max_length
from logger import logger
from utils.register import Register


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


prompt_template_ensemble = """
You are given an image and a question.
Image: {image}
Question: {question}
Given your the following candidate answers and its condifence, please select the best one.
"""
# next
"""
Answer 1: {answer1}
Confidence1: {confidence1}
Answer 2: {answer2}  
Confidence2: {confidence2}
Answer 3: {answer3}
Confidence3: {confidence3}
Answer 4: {answer4}
Confidence4: {confidence4}
My answer is:
"""


@Register(name="mpdocvqa_cot_vqa_qwen2vl_train_preprocessorv1")
class MPDocVQACoTVQAQwen2VLTrainPreprocessor(BasePreprocessor):
    def __init__(
        self,
        model_path,
        max_seq_length=1024,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        system_message="You are a helpful assistant.",
    ) -> None:
        super().__init__()

        self.template = Qwen2VLTemplate()
        self.max_seq_length = max_seq_length
        self.system_message = system_message
        self.processor = Qwen2VLProcessor.from_pretrained(
            model_path, min_pixels=min_pixels, max_pixels=max_pixels
        )

    def get_prompt(self, image, question, candidate_answers, true_answer):
        prompt = prompt_template_ensemble.format(image=image, question=question)
        # 添加候选答案部分
        for idx, (answer, confidence) in enumerate(candidate_answers):
            prompt += f"Answer {idx+1}: {answer}\nConfidence{idx+1}: {confidence}\n"
        prompt += "My answer is:"
        return self.get_openai_format(prompt, true_answer)

    def get_openai_format(self, prompt, answer):
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
        true_image_path = true_page["image_path"]
        true_ocr_path = true_page["ocr_path"]
        candidate_answers = item["candidate_answers"]

        # 从所有候选答案中选择能和正确答案匹配的答案
        answers = item["answers"]
        lower_answers = [a.strip().lower() for a in answers]
        true_candidate_answer = None
        for ca in candidate_answers:
            if ca["pred_answer"].lower().strip() in lower_answers:
                true_candidate_answer = ca
                break
        assert (
            true_candidate_answer is not None
        ), f"qid: {qid} has no true candidate answer"

        image = Image.open(true_image_path).convert("RGB")
        train_conversation, test_conversation = self.get_prompt(
            image=self.template.image_placeholder,
            question=question,
            candidate_answers=[(ca['pred_answer'], ca['confidence']) for ca in candidate_answers],
            true_answer=true_candidate_answer["pred_answer"],
        )
        train_text = self.get_text(train_conversation, add_generation_prompt=False)
        # test_text = self.get_text(test_conversation, add_generation_prompt=True)
        if check_over_max_length(
            train_text,
            max_length=self.max_seq_length,
            tokenizer=self.processor.tokenizer,
            image_processor=self.processor.image_processor,
        ):
            logger.warning(f"qid: {qid} over max length: {self.max_seq_length}")
        train_inputs = self.processor(
            text=[train_text],
            images=[image],
            videos=None,
            padding="max_length",
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        train_labels = generate_labels(
            train_inputs["input_ids"],
            [train_text],
            self.processor.tokenizer,
            self.processor.image_processor,
            self.template,
            replace_text=True,
            image_grid_thw=train_inputs["image_grid_thw"],
        )
        model_inputs = dict()
        extra = dict(
            qid=qid,
            true_image_path=true_image_path,
            true_ocr_path=true_ocr_path,
            answers=answers,
            candidate_answers=candidate_answers,
            documents=documents,
            train_conversation=train_conversation,
        )
        model_inputs.update(
            dict(
                extra=extra,
                pixel_values=train_inputs["pixel_values"],
                image_grid_thw=train_inputs["image_grid_thw"].squeeze(),
                input_ids=train_inputs["input_ids"].squeeze(),
                attention_mask=train_inputs["attention_mask"].squeeze(),
                labels=train_labels.squeeze(),
            )
        )
        return model_inputs
    

@Register(name="mpdocvqa_cot_vqa_qwen2vl_test_preprocessorv1")
class MPDocVQACoTVQAQwen2VLTestPreprocessor(BasePreprocessor):
    def __init__(
        self,
        model_path,
        classify_result_path:str,
        max_seq_length=1024,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        system_message="You are a helpful assistant.",
    ) -> None:
        super().__init__()

        self.template = Qwen2VLTemplate()
        self.max_seq_length = max_seq_length
        self.system_message = system_message
        self.processor = Qwen2VLProcessor.from_pretrained(
            model_path, min_pixels=min_pixels, max_pixels=max_pixels
        )
        self.qid2classifyitems = self.groupby_classify_result(classify_result_path)
        
    def groupby_classify_result(self, classify_result_path):
        qid2items = defaultdict(dict)
        with open(classify_result_path, 'r', encoding='utf-8') as f:
            classify_result = json.load(f)
        for item in classify_result:
            qid = item['qid']
            page_id = item['image_path'].split('/')[-1].split('.')[0]
            qid2items[qid][page_id] = item['model_output']
        return qid2items
    
    def get_prompt(self, image, question, candidate_answers, true_answer):
        prompt = prompt_template_ensemble.format(image=image, question=question)
        # 添加候选答案部分
        for idx, (answer, confidence) in enumerate(candidate_answers):
            prompt += f"Answer {idx+1}: {answer}\nConfidence{idx+1}: {confidence}\n"
        prompt += "My answer is:"
        return self.get_openai_format(prompt, true_answer)

    def get_openai_format(self, prompt, answer):
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
        # true_page = documents[true_answer_page_idx]
        # true_image_path = true_page["image_path"]
        # true_ocr_path = true_page["ocr_path"]
        answers = item["answers"]
        candidate_answers = item["candidate_answers"]

        classify_items:dict = self.qid2classifyitems[qid]
        # 根据分类结果选择得分最高的score对应的文档
        for i, doc in enumerate(documents):
            page_id = doc['page_id']
            doc['is_true_page'] = i == true_answer_page_idx
            if page_id in classify_items:
                doc['score'] = classify_items[page_id]
            else:
                raise ValueError(f"page_id: {page_id} not found in classify_result")
        documents = sorted(documents, key=lambda x: x['score'], reverse=True)
        top1_image_path = documents[0]['image_path']
        top1_ocr_path = documents[0]['ocr_path']

        image = Image.open(top1_image_path).convert("RGB")
        train_conversation, test_conversation = self.get_prompt(
            image=self.template.image_placeholder,
            question=question,
            candidate_answers=[(ca['pred_answer'], ca['confidence']) for ca in candidate_answers],
            true_answer="fake answer",
        )
        # train_text = self.get_text(train_conversation, add_generation_prompt=False)
        test_text = self.get_text(test_conversation, add_generation_prompt=True)
        if check_over_max_length(
            test_text,
            max_length=self.max_seq_length,
            tokenizer=self.processor.tokenizer,
            image_processor=self.processor.image_processor,
        ):
            logger.warning(f"qid: {qid} over max length: {self.max_seq_length}")
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
            question=question,
            answers=answers,
            candidate_answers=candidate_answers,
            documents=documents,            
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
