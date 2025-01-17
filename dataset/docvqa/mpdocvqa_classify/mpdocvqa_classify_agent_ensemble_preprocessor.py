"""
    模型集成训练
"""

import random
import json
from typing import List
from PIL import Image
import torch

import torch.nn.functional as F


from transformers import Qwen2VLProcessor
from dataset.qwen2vl.template import Qwen2VLTemplate
from dataset.base_preprocessor import BasePreprocessor
from dataset.qwen2vl.preprocess import generate_labels
from logger import logger
from utils.register import Register

prompt_template_for_classify_ensemble = """You are given an image, a question and some agent prediction score.
Image: {image}
Question: {question}
"""

# example
prompt_template_for_classify_example = """You are given an image and a question. 
Image: {image}
Question: {question}
Agent1_score: {score1}
Agent2_score: {score2}
if you can get the answer from the image, please input 'A', otherwise, please input 'B'."""


@Register(name="mpdocvqa_classify_qwen2vl_ensemble_preprocessor")
class MPDocVQAClassifyQwen2VLPreprocessor(BasePreprocessor):
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

    def get_openai_format(self, prompt, answer):
        train_ret = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
        test_ret = [{"role": "user", "content": prompt}]
        return train_ret, test_ret

    def get_prompt(self, image, question, classify_scores: List[float], true_answer):
        prompt = prompt_template_for_classify_ensemble.format(
            image=image,
            question=question,
        )

        def normalize_score(score: float):
            if score > 1:
                score = F.sigmoid(torch.tensor(score))
                score = score.item()
            return score

        # 添加候选模型得分
        for i, score in enumerate(classify_scores):
            prompt += f"Agent{i + 1}_score: {normalize_score(score):.4f}\n"
        # 添加最后的提示
        prompt += "if you can get the answer from the image, please input 'A', otherwise, please input 'B'."
        return self.get_openai_format(prompt, true_answer)

    def get_text(self, messages, add_generation_prompt=False):
        """
        将对话转为添加了特殊标记的文本
        """
        self.template.clear_messages()
        # self.add_generation_prompt = add_generation_prompt
        for message in messages:
            role = message["role"]
            msg = message["content"]
            self.template.add_message(role, msg)
        return self.template.get_prompt(add_generation_prompt)

    def preprocess(self, item):

        qid = item["qid"]
        question = item["question"]
        page_id = item["page_id"]
        image_path = item["image_path"]
        ocr_path = item["ocr_path"]
        candidate_result = item["candidate_result"]
        cls_label = item.get("cls_label", 0)
        label = "B" if cls_label == 0 else "A"

        image = Image.open(image_path).convert("RGB")
        # 分类的input
        train_conversation, test_conversation = self.get_prompt(
            image=self.template.image_placeholder,
            question=question,
            classify_scores=[item["model_output"] for item in candidate_result],
            true_answer=label,
        )
        train_text = self.get_text(train_conversation, add_generation_prompt=False)
        test_text = self.get_text(test_conversation, add_generation_prompt=True)

        train_inputs = self.processor(
            text=[train_text],
            images=[image],
            videos=None,
            padding="max_length",
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        test_inputs = self.processor(
            text=[test_text],
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

        # return
        model_inputs = dict()
        extra = dict(
            qid=qid,
            image_path=image_path,
            ocr_path=ocr_path,
            classify_conversation=test_conversation,
            # vqa_train_conversation = vqa_train_conversation,
            # answers=answers,
            classify_label=cls_label,
        )
        # model_inputs.update(extra)
        # self.save_keys = list(extra.keys())
        model_inputs.update(
            dict(
                extra=extra,
                # classify for mlp, 保持不变
                pixel_values=test_inputs["pixel_values"],
                image_grid_thw=test_inputs["image_grid_thw"].squeeze(),
                input_ids=test_inputs["input_ids"].squeeze(),
                attention_mask=test_inputs["attention_mask"].squeeze(),
                cls_label=torch.tensor(cls_label, dtype=torch.long),
                # train for ab, new add
                train_pixel_values=train_inputs["pixel_values"],
                train_image_grid_thw=train_inputs["image_grid_thw"].squeeze(),
                train_input_ids=train_inputs["input_ids"].squeeze(),
                train_attention_mask=train_inputs["attention_mask"].squeeze(),
                train_labels=train_labels.squeeze(),
            )
        )
        return model_inputs
