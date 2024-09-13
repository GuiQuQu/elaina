import abc
from typing import List
from PIL import Image

import torch

from dataset.mydataset.docvqa.preprocess import (
    preprocess_internlm,
    preprocess_internlm_for_test,
    dynamic_preprocess,
    build_transform,
)
from mpdocvqa_classify.internvl2_for_classify.internvl2.tokenization_internlm2 import (
    InternLM2Tokenizer,
)
from logger import logger

class BasePreprocessor(abc.ABC):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def preprocess(self, item):
        pass


prompt_template = """
You are given an image and a question. You need to answer the question based on the image.
Image: {image}
Question: {question}
IF xxx
"""


class InternVL2Preprocessor(BasePreprocessor):
    def __init__(
        self,
        model_path,
        num_image_token=256,
        template_name="internlm_chat",
        dynamic_image_size=True,
        use_thumbnail=True,
        min_dynamic_patch=1,
        max_dynamic_patch=6,
        pad2square=False,
        max_seq_length=1024,
        system_message='你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。',
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

        logger.info(f'[Preprocessor] num_image_token: {num_image_token}')
        logger.info(f'[Preprocessor] dynamic_image_size: {dynamic_image_size}')
        logger.info(f'[Preprocessor] use_thumbnail: {use_thumbnail}')
        logger.info(f'[Preprocessor] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}')

        self.tokenizer: InternLM2Tokenizer = InternLM2Tokenizer.from_pretrained(model_path)
        self.tokenizer.model_max_length = max_seq_length
        # image
        self.train_transform = build_transform(is_train=True, input_size=448, pad2square=pad2square)
        self.test_transform = build_transform(is_train=False, input_size=448, pad2square=pad2square)

    def get_prompt(self, label,**kwargs):
        prompt = prompt_template.format(**kwargs)
        train_ret = [
            {'from': 'human', 'value': prompt},
            {'from': 'gpt', 'value': label}
        ]
        test_ret = [
            {'from': 'human', 'value': prompt}
        ]
        return train_ret, test_ret
    
    # def get_pixel_values(self, image_path_list:List[str], transform):
    #     images, num_tiles = [], []

    def get_pixel_values(self,image_path, transform):
        image = Image.open(image_path).convert("RGB")
        if self.dynamic_image_size:
            images = dynamic_preprocess(
                image=image,
                min_num=self.min_dynamic_patch,
                max_num=self.max_dynamic_patch,
                image_size = 448,
                use_thumbnail=self.use_thumbnail
            )
            pixel_values = [transform(img) for img in images]
            pixel_values = torch.stack(pixel_values) # [N,3,448,448]
        else:
            pixel_values = transform(image)
        num_tiles = [pixel_values.size(0)] # N , meaning the number of patches(448 * 448)
        return pixel_values, num_tiles

    def preprocess(self, item):
        qid = item["qid"]
        question = item["question"]
        image_path = item["image_path"]
        ocr_path = item["ocr_path"]
        cls_label = item.get("cls_label", 0) 

        pixel_values, num_tiles = self.get_pixel_values(image_path, self.train_transform)
        test_pixel_values, _ = self.get_pixel_values(image_path, self.test_transform)
        train_conversation, test_conversation = self.get_prompt(
            cls_label = cls_label, 
            image="<image>", 
            question=question
        )
        train_inputs = preprocess_internlm(
            template_name=self.template_name,
            sources=[train_conversation],
            tokenizer=self.tokenizer,
            num_image_token_list=[self.num_image_token*num_tile for num_tile in num_tiles],
            text_only=False,
            ds_name='train_mpdocvqa',
            system_message=self.system_message,
            num_image=1
        )
        test_inputs = preprocess_internlm_for_test(
            template_name=self.template_name,
            sources=[test_conversation],
            tokenizer=self.tokenizer,
            num_image_token_list=[self.num_image_token*num_tile for num_tile in num_tiles],
            text_only=False,
            ds_name='test_mpdocvqa',
            system_message=self.system_message,
            num_image=1
        )
        model_inputs = dict(
            pixel_values=pixel_values,
            test_pixel_values=test_pixel_values,
            input_ids = train_inputs['input_ids'].squeeze(),
            attention_mask = train_inputs['attention_mask'].squeeze(),
            image_flags = torch.tensor([1] * pixel_values.size(0),dtype=torch.long),
            cls_label = torch.tensor(cls_label,dtype=torch.long),
            test_input_ids = test_inputs['input_ids'].squeeze(),
            test_attention_mask = test_inputs['attention_mask'].squeeze(),
        )
        return model_inputs