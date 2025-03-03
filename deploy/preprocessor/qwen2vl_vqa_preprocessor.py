import random
from PIL import Image
import torch
import os

from transformers import Qwen2VLProcessor, Qwen2Tokenizer
from dataset.qwen2vl.template import Qwen2VLTemplate
from dataset.base_preprocessor import BasePreprocessor
from dataset.docvqa.ocr2layout.mp_ocr2layout import (
    transform_ocr2layout as mp_transform_ocr2layout,
)
from dataset.docvqa.ocr2layout.sp_ocr2layout import (
    transform_ocr2layout as sp_transform_ocr2layout,
)
from dataset.docvqa.docvqa_utils import truncate_layout
from logger import logger
from utils.register import Register


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


@Register(name="gradio_vqa_qwen2vl_preprocessor")
class GradioVQAQwen2VLPreprocessor(BasePreprocessor):
    def __init__(
        self,
        model_path,
        use_ocr=False,
        max_layout_length=1024,
        max_seq_length= 2048,
        ocr_dir: str = None,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        system_message="You are a helpful assistant.",
    ) -> None:
        super().__init__()

        self.template = Qwen2VLTemplate()
        self.system_message = system_message
        self.processor = Qwen2VLProcessor.from_pretrained(
            model_path, min_pixels=min_pixels, max_pixels=max_pixels
        )
        self.use_ocr = use_ocr
        self.max_layout_length = max_layout_length
        self.ocr_dir = ocr_dir

        self.max_seq_length = max_seq_length
        if self.use_ocr:
            assert self.ocr_dir is not None, "OCR dir is required when use_ocr is True"

    def get_prompt(self, prompt_template, answer, **kwargs):
        prompt = prompt_template.format(**kwargs)
        # 按照openai的格式组织输入
        train_ret = [
            {
                "role": "user",
                "content":prompt,
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ]
        test_ret = [
            {
                "role": "user",
                "content":prompt,
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
        """
        /home/klwang/data/MPDocVQA/ocr/abc.json
        """
        if not os.path.exists(ocr_path):
            return None
        dataset_name = ocr_path.split("/")[-3].lower().strip()
        if dataset_name == "spdocvqa":
            layout = sp_transform_ocr2layout(ocr_path)
        elif dataset_name == "mpdocvqa":
            layout = mp_transform_ocr2layout(ocr_path)
        layout, is_truncated = truncate_layout(
            layout,
            tokenizer=self.processor.tokenizer,
            max_token_length=self.max_layout_length,
        )
        if is_truncated:
            ocr_name = ocr_path.split("/")[-1]
            logger.info(f"[{self.__class__.__name__}] layout({ocr_name}) is truncated")
        return layout

    def preprocess(self, item):

        question = item["question"]
        image_path = item["image_path"]
        page_id = image_path.split("/")[-1].split(".")[0]
        ocr_path = f"{self.ocr_dir}/{page_id}.json"
        image = Image.open(image_path).convert("RGB")

        layout = self.transform_ocr2layout(ocr_path)
        prompt_template_local = (
            prompt_template_add_layout if self.use_ocr else prompt_template
        )

        _, test_conversation = self.get_prompt(
            prompt_template_local,
            answer="fake answer",
            image=self.template.image_placeholder,
            question=question,
            layout=layout if self.use_ocr else None,
        )
        # train_text = self.get_text(train_conversation, add_generation_prompt=False)
        test_text = self.get_text(test_conversation, add_generation_prompt=True)

        test_inputs = self.processor(
            text=[test_text],
            images=[image],
            videos=None,
            # padding="max_length",
            # max_length=self.max_seq_length,
            return_tensors="pt",
        )
        model_inputs = dict()
        extra = dict(
            question=question,
            image_path=image_path,
            ocr_path=ocr_path,
            test_conversation=test_conversation,
        )
        model_inputs.update(
            dict(
                extra=extra,
                test_pixel_values=test_inputs["pixel_values"],
                test_image_grid_thw=test_inputs["image_grid_thw"],
                test_input_ids=test_inputs["input_ids"],
                test_attention_mask=test_inputs["attention_mask"],
            )
        )
        return model_inputs
