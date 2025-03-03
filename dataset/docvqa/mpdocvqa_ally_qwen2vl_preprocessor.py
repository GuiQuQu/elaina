"""
    训练单图任务，根据给定的true_answer_page_idx，只让模型在根据这个page_id的图像回答问题
"""

import random
import json
from PIL import Image
import torch

from transformers import Qwen2VLProcessor, Qwen2Tokenizer
from dataset.qwen2vl.template import Qwen2VLTemplate
from dataset.base_preprocessor import BasePreprocessor
from dataset.qwen2vl.preprocess import generate_labels
from logger import logger
from utils.register import Register


prompt_template_for_classify = """You are given an image and a question. 
Image: {image}
Question: {question}
if you can get the answer from the image, please input 'A', otherwise, please input 'B'."""

prompt_template_for_vqa = """You are given an image and a question. 
Image: {image}
Question: {question}
Please answer the question based on the image.
You should extract the answer from the text in the image without changing the order and form of the words.
Answer: """


@Register(name="mpdocvqa_ally_qwen2vl_preprocessor")
class MPDocVQAAllyQwen2VLPreprocessor(BasePreprocessor):
    def __init__(
        self,
        model_path,
        max_seq_length=1024,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        system_message="You are a helpful assistant.",
        fake_answer_json_path = "dataset/docvqa/no_answer/qwen.json",
    ) -> None:
        super().__init__()

        self.template = Qwen2VLTemplate()
        self.max_seq_length = max_seq_length
        self.system_message = system_message
        self.fake_answer_list = self.load_fake_answer(fake_answer_json_path)
        # self.tokenizer: Qwen2Tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        # self.tokenizer.model_max_length = max_seq_length
        self.processor = Qwen2VLProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)

    def load_fake_answer(self, json_path):
        with open(json_path, "r") as f:
            fake_answer_list = json.load(f)
        # 去重
        fake_answer_list = list(set(fake_answer_list))
        return fake_answer_list

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
        answers = item["answers"]
        if len(answers) > 0:
            answer = random.choice(answers)
        else:
            answer = random.choice(self.fake_answer_list)

        cls_label = item.get("cls_label", 0)

        # qid = item["qid"]
        # question = item["question"]
        # documents = item["documents"]
        # true_answer_page_idx = item["true_answer_page_idx"]
        # true_page = documents[true_answer_page_idx]
        # true_image_path = true_page["image_path"]
        # true_ocr_path = true_page["ocr_path"]
        # answers = item["answers"]

        image = Image.open(image_path).convert("RGB")

        # 分类的input
        _, classify_conversation = self.get_prompt(
            prompt_template=prompt_template_for_classify,
            image=self.template.image_placeholder,
            question=question,
            answer="A" if cls_label == 1 else "B",
        )
        classify_text = self.get_text(
            classify_conversation,
            add_generation_prompt=True,
        )
        classify_inputs = self.processor(
            text = [classify_text],
            images=[image],
            videos=None,
            padding="max_length",
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        # vqa的input
        vqa_train_conversation, vqa_test_conversation = self.get_prompt(
            prompt_template=prompt_template_for_vqa,
            answer=random.choice(answers),
            image=self.template.image_placeholder,
            question=question,
        )
        vqa_train_text = self.get_text(vqa_train_conversation, add_generation_prompt=False)
        vqa_test_text = self.get_text(vqa_test_conversation, add_generation_prompt=True)
        
        vqa_train_inputs = self.processor(
            text = [vqa_train_text],
            images = [image],
            videos=None,
            padding="max_length",
            max_length = self.max_seq_length,
            return_tensors="pt",
        )
        vqa_labels = generate_labels(
            vqa_train_inputs["input_ids"],
            [vqa_train_text],
            self.processor.tokenizer,
            self.processor.image_processor,
            self.template,
            replace_text=True,
            image_grid_thw=vqa_train_inputs["image_grid_thw"],
        )

        vqa_test_inputs = self.processor(
            text = [vqa_test_text],
            images = [image],
            videos=None,
            padding="max_length",
            max_length = self.max_seq_length,
            return_tensors="pt",
        )

        # return
        model_inputs = dict()
        extra = dict(
            qid=qid,
            image_path=image_path,
            ocr_path=ocr_path,
            classify_conversation = classify_conversation,
            vqa_train_conversation = vqa_train_conversation,
            answers=answers,
            classify_label = cls_label,
        )
        # model_inputs.update(extra)
        # self.save_keys = list(extra.keys())
        model_inputs.update(
            dict(
                extra = extra,
                # classify
                classify_pixel_values=classify_inputs["pixel_values"],
                classify_image_grid_thw=classify_inputs["image_grid_thw"].squeeze(),
                classify_input_ids=classify_inputs["input_ids"].squeeze(),
                classify_attention_mask=classify_inputs["attention_mask"].squeeze(),
                classify_label=torch.tensor(cls_label, dtype=torch.long),
                # vqa
                vqa_pixel_values=vqa_train_inputs["pixel_values"],
                vqa_image_grid_thw=vqa_train_inputs["image_grid_thw"].squeeze(),
                vqa_input_ids=vqa_train_inputs["input_ids"].squeeze(),
                vqa_attention_mask=vqa_train_inputs["attention_mask"].squeeze(),
                vqa_label=vqa_labels.squeeze(),
                # test classify
                test_classify_pixel_values=classify_inputs["pixel_values"],
                test_classify_image_grid_thw=classify_inputs["image_grid_thw"].squeeze(),
                test_classify_input_ids=classify_inputs["input_ids"].squeeze(),
                test_classify_attention_mask=classify_inputs["attention_mask"].squeeze(),

                # test vqa
                test_vqa_pixel_values=vqa_test_inputs["pixel_values"],
                test_vqa_image_grid_thw=vqa_test_inputs["image_grid_thw"].squeeze(),
                test_vqa_input_ids=vqa_test_inputs["input_ids"].squeeze(),
                test_vqa_attention_mask=vqa_test_inputs["attention_mask"].squeeze(),
            )
        )
        return model_inputs
