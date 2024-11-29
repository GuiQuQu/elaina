"""
    训练单图任务，根据给定的true_answer_page_idx，只让模型在根据这个page_id的图像回答问题
"""

import random
from PIL import Image
import torch

from models.docvqa.qwenvl.tokenization_qwen import QWenTokenizer

from dataset.base_preprocessor import BasePreprocessor
from logger import logger
from dataset.docvqa.ocr2layout.sp_ocr2layout import transfrom_ocr2plain_text
from dataset.docvqa.docvqa_utils import truncate_layout_by_length
from dataset.qwenvl.preprocess_qwenvl import generate_labels_for_qwenvl
from dataset.qwenvl.template import QwenVLTemplate
from utils.register import Register

# prompt_template = """You are given an image and a question. 
# Image: {image}
# Question: {question}
# Please answer the question based on the image.
# You should extract the answer from the text in the image without changing the order and form of the words.
# Answer: """

# prompt_template_add_layout = """You are given an image,its corresponding string layout and a question.
# Image: {image}
# String Layout: 
# {layout}
# Question: {question}
# Please answer the question based on the image and its its corresponding string layout.
# You should extract the answer from the text in the image without changing the order and form of the words.
# Answer:"""

prompt_template_for_qwenvl_add_layout= """You are asked to answer questions based on the given document image and its corresponding string layout. The layout and image is included by "```".
The answers to questions are short text spans token verbatim from the layout or image.This means answers comprise a set of contiguous text tokens present in the layout or image.
Document Picture:
```
{image_path}
```
Document:
```
{layout}
```
Question: {question}
Directly extrct the answer of the question from the document layout and image with as few words as possible.
Answer:
"""

@Register(name="docvqa_vqa_qwenvl_plain_ocr_preprocessor")
class DocVQAVqaQwenVLPlainOCRPreprocessor(BasePreprocessor):
    def __init__(
        self,
        model_path,
        max_layout_length=1024,
        max_seq_length=2048,
        system_message="You are a helpful assistant.",
    ) -> None:
        super().__init__()

        self.max_seq_length = max_seq_length
        self.max_layout_length = max_layout_length
        self.system_message = system_message
        self.template = QwenVLTemplate()
        self.template.set_system_message(system_message)

        self.tokenizer: QWenTokenizer = QWenTokenizer.from_pretrained(
            model_path
        )
        self.tokenizer.model_max_length = max_seq_length
        self.tokenizer.pad_token_id = self.tokenizer.eod_id

    def get_prompt(self,prompt_template, answer:str, **kwargs):
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


    def transform_ocr2plain_text(self, ocr_path):
        layout = transfrom_ocr2plain_text(ocr_path)
        layout, is_truncated = truncate_layout_by_length(
            layout, tokenizer=self.tokenizer, max_token_length=self.max_layout_length
        )
        return layout

    def preprocess(self, item):

        qid = item["qid"]
        question = item["question"]
        image_path = item["image_path"]
        ocr_path = item["ocr_path"]
        answers = item["answers"]

        layout = self.transform_ocr2plain_text(ocr_path)
        train_conversation, test_conversation = self.get_prompt(
            prompt_template=prompt_template_for_qwenvl_add_layout,
            answer=random.choice(answers),
            image_path=f"<img>{image_path}</img>",
            question=question,
            layout=layout,
        )
        train_text = self.get_text(train_conversation, add_generation_prompt=False)
        test_text = self.get_text(test_conversation,add_generation_prompt=True)
    
        train_inputs = self.tokenizer(
            train_text, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_seq_length
        )
        train_labels = generate_labels_for_qwenvl(
            input_ids = train_inputs["input_ids"], 
            texts = [train_text],
            tokenizer=self.tokenizer,
            template=self.template,
        )

        # 推理用数据left padding
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        test_inputs = self.tokenizer(
            test_text, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_seq_length
        )
        self.tokenizer.padding_side = original_padding_side
        
        model_inputs = dict()
        extra = dict(
            qid=qid,
            image_path=image_path,
            ocr_path=ocr_path,
            answers=answers,
            # train_conversation=train_conversation,
            test_conversation=test_conversation,
        )
        # model_inputs.update(extra)
        # self.save_keys = list(extra.keys())
        model_inputs.update(
            dict(
                extra = extra,
                input_ids=train_inputs["input_ids"].squeeze(),
                attention_mask=train_inputs["attention_mask"].squeeze(),
                labels=train_labels.squeeze(),
                # test_conversation=test_conversation,

                test_input_ids = test_inputs["input_ids"].squeeze(),
                test_attention_mask = test_inputs["attention_mask"].squeeze(),
            )
        )
        return model_inputs
