"""
    训练单图任务，根据给定的true_answer_page_idx，只让模型在根据这个page_id的图像回答问题
"""

import random
from PIL import Image
import torch

from transformers import Qwen2VLProcessor, Qwen2Tokenizer
from dataset.qwen2vl.template import Qwen2VLTemplate
from dataset.base_preprocessor import BasePreprocessor
from dataset.qwen2vl.preprocess import generate_labels
from logger import logger
from utils.register import Register
from dataset.docvqa.ocr2layout.mp_ocr2layout import transform_ocr2layout
from dataset.docvqa.docvqa_utils import truncate_layout

# prompt_template = """You are given an image and a question. 
# Image: {image}
# Question: {question}
# Please answer the question based on the image.
# You should extract the answer from the text in the image without changing the order and form of the words.
# Answer: """

prompt_template_add_layout = """You are given an image,its corresponding string layout and a question.
Image: {image}
String Layout: 
{layout}
Question: {question}
Please answer the question based on the image and its its corresponding string layout.
You should extract the answer from the text in the image without changing the order and form of the words.
Answer:"""


@Register(name="mpdocvqa_vqa_ocr_qwen2vl_preprocessor")
class MPDocVQAVQAOCRQwen2VLPreprocessor(BasePreprocessor):
    def __init__(
        self,
        model_path,
        max_seq_length=1024,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        max_layout_length=960,
        system_message="You are a helpful assistant.",
    ) -> None:
        super().__init__()

        self.template = Qwen2VLTemplate()
        self.max_seq_length = max_seq_length
        self.system_message = system_message
        self.max_layout_length = max_layout_length
        # self.tokenizer: Qwen2Tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        # self.tokenizer.model_max_length = max_seq_length
        self.processor = Qwen2VLProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)

    def transform_ocr2layout(self, ocr_path):
        layout = transform_ocr2layout(ocr_path)
        layout, is_truncated = truncate_layout(
            layout, tokenizer=self.processor.tokenizer, max_token_length=self.max_layout_length
        )
        if (is_truncated):
            ocr_name = ocr_path.split("/")[-1]
            logger.info(f"[{self.__class__.__name__}] layout({ocr_name}) is truncated")
        return layout

    def get_prompt(self, answer, **kwargs):
        prompt = prompt_template_add_layout.format(**kwargs)
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
        documents = item["documents"]
        true_answer_page_idx = item["true_answer_page_idx"]
        true_page = documents[true_answer_page_idx]
        true_image_path = true_page["image_path"]
        true_ocr_path = true_page["ocr_path"]
        answers = item["answers"]

        image = Image.open(true_image_path).convert("RGB")
        
        layout = self.transform_ocr2layout(true_ocr_path)
        if layout.strip() == "":
            logger.warning(f"[{self.__class__.__name__}] layout is empty, qid: {qid}")
        train_conversation, test_conversation = self.get_prompt(
            answer=random.choice(answers),
            image=self.template.image_placeholder,
            question=question,
            layout = layout,
        )
        train_text = self.get_text(train_conversation, add_generation_prompt=False)
        # test_text = self.get_text(test_conversation, add_generation_prompt=True)
        train_inputs = self.processor(
            text = [train_text],
            images = [image],
            videos=None,
            padding="max_length",
            max_length = self.max_seq_length,
            return_tensors="pt",
        )

        # test_inputs = self.processor(
        #     text = [test_text],
        #     images = [image],
        #     videos=None,
        #     padding="max_length",
        #     max_length = self.max_seq_length,
        #     return_tensors="pt",
        # )

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
            documents=documents,
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
                # test_pixel_values=test_inputs["pixel_values"],
                # test_image_grid_thw=test_inputs["image_grid_thw"].squeeze(),
                # test_input_ids=test_inputs["input_ids"].squeeze(),
                # test_attention_mask=test_inputs["attention_mask"].squeeze(),
            )
        )
        return model_inputs
