"""
    训练单图任务，根据给定的true_answer_page_idx，只让模型在根据这个page_id的图像回答问题
"""

import random
from PIL import Image
import torch

from transformers import Qwen2VLProcessor
from dataset.qwen2vl.template import Qwen2VLTemplate
from dataset.base_preprocessor import BasePreprocessor
from dataset.qwen2vl.preprocess import generate_labels
from logger import logger
from utils.register import Register

from dataset.docvqa.ocr2layout.sp_ocr2layout import transform_ocr2layout
from dataset.docvqa.docvqa_utils import truncate_layout

prompt_template = """You are given an image and a question. 
Image: {image}
Question: {question}
Please answer the question based on the image.
You should extract the answer from the text in the image without changing the order and form of the words.
Answer: """

prompt_template_add_layout = """You are given an image,its corresponding string layout and a question.
Image: {image}
String Layout: 
{layout}
Question: {question}
Please answer the question based on the image and its its corresponding string layout.
You should extract the answer from the text in the image without changing the order and form of the words.
Answer:"""

@Register(name="docvqa_vqa_qwen2vl_preprocessor")
class DocVQAVQAQwen2VLPreprocessor(BasePreprocessor):
    def __init__(
        self,
        model_path,
        use_ocr = False,
        max_layout_length=1024,
        max_seq_length=1024,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        system_message="You are a helpful assistant.",
    ) -> None:
        super().__init__()

        self.template = Qwen2VLTemplate()
        self.max_seq_length = max_seq_length
        self.system_message = system_message
        
        # 
        self.use_ocr = use_ocr
        self.max_layout_length = max_layout_length
        self.prompt_template = prompt_template_add_layout if use_ocr else prompt_template

        # self.tokenizer: Qwen2Tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        # self.tokenizer.model_max_length = max_seq_length
        self.processor = Qwen2VLProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)

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

    def transform_ocr2layout(self, ocr_path):
        layout = transform_ocr2layout(ocr_path, placeholder=" ")
        layout, is_truncated = truncate_layout(
            layout, tokenizer=self.processor.tokenizer, max_token_length=self.max_layout_length
        )
        return layout

    def preprocess(self, item):

        qid = item["qid"]
        question = item["question"]
        image_path = item["image_path"]
        ocr_path = item['ocr_path']
        answers = item["answers"]

        image = Image.open(image_path).convert("RGB")
        
        layout = self.transform_ocr2layout(ocr_path)
        train_conversation, test_conversation = self.get_prompt(
            self.prompt_template,
            answer=random.choice(answers),
            image=self.template.image_placeholder,
            question=question,
            layout = layout if self.use_ocr else None,
        )
        train_text = self.get_text(train_conversation, add_generation_prompt=False)
        test_text = self.get_text(test_conversation, add_generation_prompt=True)
        
        train_inputs = self.processor(
            text = [train_text],
            images = [image],
            videos=None,
            padding="max_length",
            max_length = self.max_seq_length,
            return_tensors="pt",
        )
        if train_inputs['input_ids'].size(-1) != self.max_seq_length:
            logger.warning(f"input_ids size: {train_inputs['input_ids'].size()} is not equal to max_seq_length: {self.max_seq_length}")
        train_labels = generate_labels(
            train_inputs["input_ids"],
            [train_text],
            self.processor.tokenizer,
            self.processor.image_processor,
            self.template,
            replace_text=True,
            image_grid_thw=train_inputs["image_grid_thw"],
        )
        test_inputs = self.processor(
            text = [test_text],
            images = [image],
            videos=None,
            padding="max_length",
            max_length = self.max_seq_length,
            return_tensors="pt",
        )
        model_inputs = dict()
        extra = dict(
            qid=qid,
            image_path=image_path,
            ocr_path=ocr_path,
            answers=answers,
            test_conversation=test_conversation,
        )
        # model_inputs.update(extra)
        # self.save_keys = list(extra.keys())
        model_inputs.update(
            dict(
                extra = extra,
                pixel_values=train_inputs["pixel_values"],
                image_grid_thw=train_inputs["image_grid_thw"].squeeze(),
                input_ids=train_inputs["input_ids"].squeeze(),
                attention_mask=train_inputs["attention_mask"].squeeze(),
                labels=train_labels.squeeze(),
                
                # test input 
                test_pixel_values=test_inputs["pixel_values"],
                test_image_grid_thw=test_inputs["image_grid_thw"].squeeze(),
                test_input_ids=test_inputs["input_ids"].squeeze(),
                test_attention_mask=test_inputs["attention_mask"].squeeze(),
            )
        )
        return model_inputs
