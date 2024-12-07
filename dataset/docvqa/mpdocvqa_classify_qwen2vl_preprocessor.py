"""
    训练单图任务，根据给定的true_answer_page_idx，只让模型在根据这个page_id的图像回答问题
"""

import random
import json
from PIL import Image
import torch

from transformers import Qwen2VLProcessor
from dataset.qwen2vl.template import Qwen2VLTemplate
from dataset.base_preprocessor import BasePreprocessor
from dataset.qwen2vl.preprocess import generate_labels
from logger import logger
from utils.register import Register


prompt_template_for_classify = """You are given an image and a question. 
Image: {image}
Question: {question}
if you can get the answer from the image, please input 'A', otherwise, please input 'B'."""

# prompt_template_for_vqa = """You are given an image and a question. 
# Image: {image}
# Question: {question}
# Please answer the question based on the image.
# You should extract the answer from the text in the image without changing the order and form of the words.
# Answer: """


@Register(name="mpdocvqa_classify_qwen2vl_preprocessor")
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
        self.processor = Qwen2VLProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)


    def get_prompt(self, prompt_template, answer, **kwargs):
        prompt = prompt_template.format(**kwargs)
        # 按照openai的格式组织输入
        train_ret = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ]
        test_ret = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        return train_ret, test_ret

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
        cls_label = item.get("cls_label", 0)
        label = "B" if cls_label == 0 else "A"

        image = Image.open(image_path).convert("RGB")
        # 分类的input
        train_conversation, test_conversation = self.get_prompt(
            prompt_template=prompt_template_for_classify,
            image=self.template.image_placeholder,
            question=question,
            answer=label,
        )
        train_text = self.get_text(train_conversation,add_generation_prompt=False)
        test_text = self.get_text(test_conversation,add_generation_prompt=True)

        train_inputs = self.processor(
            text=[train_text],
            images=[image],
            videos=None,
            padding="max_length",
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        test_inputs = self.processor(
            text = [test_text],
            images=[image],
            videos=None,
            padding="max_length",
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        train_labels = generate_labels(
            train_inputs['input_ids'],
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
            classify_conversation = test_conversation,
            # vqa_train_conversation = vqa_train_conversation,
            # answers=answers,
            classify_label = cls_label,
        )
        # model_inputs.update(extra)
        # self.save_keys = list(extra.keys())
        model_inputs.update(
            dict(
                extra = extra,
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
