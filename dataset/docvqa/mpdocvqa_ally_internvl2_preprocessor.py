import random
import json
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
from logger import logger

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


class MPDocVQAAllyInternVL2Preprocessor(BasePreprocessor):
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
        fake_answer_json_path="dataset/docvqa/no_answer/qwen.json",
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
        self.system_message = system_message
        self.fake_answer_list = self.load_fake_answer(fake_answer_json_path)
        logger.info(f"[Preprocessor] num_image_token: {num_image_token}")
        logger.info(f"[Preprocessor] dynamic_image_size: {dynamic_image_size}")
        logger.info(f"[Preprocessor] use_thumbnail: {use_thumbnail}")
        logger.info(
            f"[Preprocessor] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}"
        )
        logger.info(f"[Preprocessor] fake_answer_cnt: {len(self.fake_answer_list)}")

        self.tokenizer: InternLM2Tokenizer = InternLM2Tokenizer.from_pretrained(
            model_path
        )
        self.tokenizer.model_max_length = max_seq_length
        # image
        self.train_transform = build_transform(
            is_train=True, input_size=448, pad2square=pad2square
        )
        self.test_transform = build_transform(
            is_train=False, input_size=448, pad2square=pad2square
        )

    def load_fake_answer(self, json_path):
        with open(json_path, "r") as f:
            fake_answer_list = json.load(f)
        # 去重
        fake_answer_list = list(set(fake_answer_list))
        return fake_answer_list

    def get_prompt(self, prompt_template, label, **kwargs):
        prompt = prompt_template.format(**kwargs)
        train_ret = [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": label},
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

        # 无论train还是test,都不对image做变化
        pixel_values, num_tiles = self.get_pixel_values(
            image_path, self.train_transform
        )
        test_pixel_values, test_num_tiles = self.get_pixel_values(
            image_path, self.test_transform
        )

        # 分类text
        _, classify_conversation = self.get_prompt(
            label=cls_label,
            prompt_template=prompt_template_for_classify,
            image="<image>",
            question=question,
        )

        # vqa text for train or test
        vqa_train_conversation, vqa_test_conversation = self.get_prompt(
            label=answer,
            prompt_template=prompt_template_for_vqa,
            image="<image>",
            question=question,
        )

        classify_inputs = preprocess_internlm_for_test(
            template_name=self.template_name,
            sources=[classify_conversation],
            tokenizer=self.tokenizer,
            num_image_token_list=[
                self.num_image_token * num_tile for num_tile in num_tiles
            ],
            text_only=False,
            ds_name="classify_mpdocvqa",
            system_message=self.system_message,
            num_image=1,
        )

        vqa_train_inputs = preprocess_internlm(
            template_name=self.template_name,
            sources=[vqa_train_conversation],
            tokenizer=self.tokenizer,
            num_image_token_list=[
                self.num_image_token * num_tile for num_tile in num_tiles
            ],
            text_only=False,
            ds_name="vqa_train_mpdocvqa",
            system_message=self.system_message,
            num_image=1,
        )
        model_inputs = dict()
        extra = dict(
            qid=qid,
            image_path=image_path,
            ocr_path=ocr_path,
            classify_conversation=classify_conversation,
            vqa_train_conversation=vqa_train_conversation,
            answers=answers,
            classify_label=cls_label,
        )
        # model_inputs.update(extra)
        # 在num_worker > 0 的情况下,这行代码可能会不起作用
        # self.save_keys = list(extra.keys())
        model_inputs.update(
            dict(
                extra = extra,
                pixel_values=pixel_values,
                test_pixel_values=test_pixel_values,
                vqa_input_ids=vqa_train_inputs["input_ids"].squeeze(),
                vqa_attention_mask=vqa_train_inputs["attention_mask"].squeeze(),
                image_flags=torch.tensor([1] * pixel_values.size(0), dtype=torch.long),
                classify_input_ids=classify_inputs["input_ids"].squeeze(),
                classify_attention_mask=classify_inputs["attention_mask"].squeeze(),
                cls_label=torch.tensor(cls_label, dtype=torch.long),
                vqa_train_conversation=vqa_train_conversation,
                vqa_label=vqa_train_inputs["labels"].squeeze(),
                num_tiles=num_tiles[0],  # 单图，因此只选择第一个图像被划分出来的patch数量,for test
            )
        )
        return model_inputs
