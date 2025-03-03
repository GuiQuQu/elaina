"""
    训练单图任务，根据给定的true_answer_page_idx，只让模型在根据这个page_id的图像回答问题
"""

from collections import defaultdict
import json
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


@Register(name="mpdocvqa_vqa_ocr_qwen2vl_test_preprocessor")
class MPDocVQAVQAOCRQwen2VLTestPreprocessor(BasePreprocessor):
    def __init__(
        self,
        model_path,
        classify_result_path: str,
        max_seq_length=1024,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        max_layout_length=960,
        reverse=True,
        system_message="You are a helpful assistant.",
    ) -> None:
        super().__init__()

        self.template = Qwen2VLTemplate()
        self.max_seq_length = max_seq_length
        self.max_layout_length = max_layout_length
        # self.system_message = system_message
        self.template.set_system_message(system_message)

        # self.tokenizer: Qwen2Tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        # self.tokenizer.model_max_length = max_seq_length
        self.processor = Qwen2VLProcessor.from_pretrained(
            model_path, min_pixels=min_pixels, max_pixels=max_pixels
        )
        self.qid2classifyitems = self.groupby_classify_result(classify_result_path)
        self.reverse = reverse
        logger.info(
            f"min_tokens {min_pixels // 28 // 28}, max_tokens {max_pixels // 28 // 28}"
        )

    def transform_ocr2layout(self, ocr_path):
        layout = transform_ocr2layout(ocr_path)
        layout, is_truncated = truncate_layout(
            layout,
            tokenizer=self.processor.tokenizer,
            max_token_length=self.max_layout_length,
        )
        if is_truncated:
            ocr_name = ocr_path.split("/")[-1]
            logger.info(f"[{self.__class__.__name__}] layout({ocr_name}) is truncated")
        return layout

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
        prompt = prompt_template_add_layout.format(**kwargs)
        # 按照openai的格式组织输入
        train_ret = [
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ]
        test_ret = [
            {
                "role": "user",
                "content": prompt,
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

        image = Image.open(top1_image_path).convert("RGB")
        layout = self.transform_ocr2layout(top1_ocr_path)
        train_conversation, test_conversation = self.get_prompt(
            answer=random.choice(answers),
            image=self.template.image_placeholder,
            question=question,
            layout=layout,
        )

        # train_text = self.get_text(train_conversation, add_generation_prompt=False)
        test_text = self.get_text(test_conversation, add_generation_prompt=True)
        # train_inputs = self.processor(
        #     text=[train_text],
        #     images=[image],
        #     videos=None,
        #     padding="max_length",
        #     max_length=self.max_seq_length,
        #     return_tensors="pt",
        # )
        # train_labels = generate_labels(
        #     train_inputs["input_ids"],
        #     [train_text],
        #     self.processor.tokenizer,
        #     self.processor.image_processor,
        #     self.template,
        #     replace_text=True,
        #     image_grid_thw=train_inputs["image_grid_thw"],
        # )

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
        # model_inputs.update(extra)
        # self.save_keys = list(extra.keys())
        model_inputs.update(
            dict(
                # pixel_values=train_inputs["pixel_values"],
                # image_grid_thw=train_inputs["image_grid_thw"].squeeze(),
                # input_ids=train_inputs["input_ids"].squeeze(),
                # attention_mask=train_inputs["attention_mask"].squeeze(),
                # labels=train_labels.squeeze(),
                extra=extra,
                test_pixel_values=test_inputs["pixel_values"],
                test_image_grid_thw=test_inputs["image_grid_thw"].squeeze(),
                test_input_ids=test_inputs["input_ids"].squeeze(),
                test_attention_mask=test_inputs["attention_mask"].squeeze(),
            )
        )
        return model_inputs
