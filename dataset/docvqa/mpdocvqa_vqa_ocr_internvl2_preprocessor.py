"""
    训练单图任务，根据给定的true_answer_page_idx，只让模型在根据这个page_id的图像回答问题
"""

import random
from PIL import Image
import torch

from models.docvqa.internvl2.tokenization_internlm2 import InternLM2Tokenizer
from dataset.docvqa.preprocess import (
    build_transform,
    preprocess_internlm,
    preprocess_internlm_for_test,
    dynamic_preprocess,
)
from dataset.base_preprocessor import BasePreprocessor
from dataset.docvqa.docvqa_utils import truncate_layout
from dataset.docvqa.ocr2layout.mp_ocr2layout import transform_ocr2layout
from logger import logger

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


class MPDocVQAVQAOCRInternVL2Preprocessor(BasePreprocessor):
    def __init__(
        self,
        model_path,
        num_image_token=256,
        template_name="internlm2-chat",
        dynamic_image_size=True,
        use_thumbnail=True,
        min_dynamic_patch=1,
        max_dynamic_patch=6,
        pad2square=False,
        max_seq_length=1024,
        max_layout_length=1024,
        system_message="你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。",
    ) -> None:
        super().__init__()

        self.num_image_token = num_image_token
        self.template_name = template_name
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.pad2square = pad2square
        self.max_seq_length = max_seq_length
        self.max_layout_length = max_layout_length
        self.system_message = system_message

        logger.info(f"[Preprocessor] num_image_token: {num_image_token}")
        logger.info(f"[Preprocessor] dynamic_image_size: {dynamic_image_size}")
        logger.info(f"[Preprocessor] use_thumbnail: {use_thumbnail}")
        logger.info(
            f"[Preprocessor] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}"
        )

        self.tokenizer: InternLM2Tokenizer = InternLM2Tokenizer.from_pretrained(
            model_path
        )
        self.tokenizer.model_max_length = max_seq_length
        # image
        # sft训练也不考虑做图像增强变换
        self.train_transform = build_transform(
            is_train=False, input_size=448, pad2square=pad2square
        )
        self.test_transform = build_transform(
            is_train=False, input_size=448, pad2square=pad2square
        )

    def get_prompt(self, answer, **kwargs):
        prompt = prompt_template_add_layout.format(**kwargs)
        train_ret = [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": answer},
        ]
        test_ret = [{"from": "human", "value": prompt}]
        return train_ret, test_ret

    def get_pixel_values(self, image_path, transform):
        image = Image.open(image_path).convert("RGB")
        if self.dynamic_image_size:
            images = dynamic_preprocess(
                image=image,
                min_num=self.min_dynamic_patch,
                max_num=self.max_dynamic_patch,
                image_size=448,
                use_thumbnail=self.use_thumbnail,
            )
            pixel_values = [transform(img) for img in images]
            pixel_values = torch.stack(pixel_values)  # [N,3,448,448]
        else:
            pixel_values = transform(image)
        num_tiles = [
            pixel_values.size(0)
        ]  # N , meaning the number of patches(448 * 448)
        return pixel_values, num_tiles

    def transform_ocr2layout(self, ocr_path):
        layout = transform_ocr2layout(ocr_path)
        layout, is_truncated = truncate_layout(
            layout, tokenizer=self.tokenizer, max_token_length=self.max_layout_length
        )
        if (is_truncated):
            ocr_name = ocr_path.split("/")[-1]
            logger.info(f"[Preprocessor] layout({ocr_name}) is truncated")
        return layout

    def preprocess(self, item):

        qid = item["qid"]
        question = item["question"]
        documents = item["documents"]
        true_answer_page_idx = item["true_answer_page_idx"]
        true_page = documents[true_answer_page_idx]
        true_image_path = true_page["image_path"]
        true_ocr_path = true_page["ocr_path"]
        answers = item["answers"]
        # 选择正确page_id做单图任务回答
        pixel_values, num_tiles = self.get_pixel_values(
            true_image_path, self.train_transform
        )
        test_pixel_values, _ = self.get_pixel_values(
            true_image_path, self.test_transform
        )
        
        layout = self.transform_ocr2layout(true_ocr_path)
        train_conversation, test_conversation = self.get_prompt(
            answer=random.choice(answers),
            image="<image>",
            question=question,
            layout=layout,
        )
        train_inputs = preprocess_internlm(
            template_name=self.template_name,
            sources=[train_conversation],
            tokenizer=self.tokenizer,
            num_image_token_list=[
                self.num_image_token * num_tile for num_tile in num_tiles
            ],
            text_only=False,
            ds_name="train_mpdocvqa_vqa",
            system_message=self.system_message,
            num_image=1,
        )
        model_inputs = dict()
        # test_question = test_conversation[0]["value"]
        extra = dict(
            qid=qid,
            true_image_path=true_image_path,
            true_ocr_path=true_ocr_path,
            answers=answers,
            documents=documents,
            # train_conversation=train_conversation,
            test_conversation=test_conversation,
        )
        model_inputs.update(extra)
        self.save_keys = list(extra.keys())
        model_inputs.update(
            dict(
                pixel_values=pixel_values,
                input_ids=train_inputs["input_ids"].squeeze(),
                attention_mask=train_inputs["attention_mask"].squeeze(),
                labels=train_inputs["labels"].squeeze(),
                image_flags=torch.tensor([1] * pixel_values.size(0), dtype=torch.long),
                test_pixel_values=test_pixel_values,
                num_tiles=num_tiles[
                    0
                ],  # 单图，因此只选择第一个图像被划分出来的patch数量
            )
        )
        return model_inputs
