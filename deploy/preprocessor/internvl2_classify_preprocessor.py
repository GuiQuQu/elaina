"""
    同时指出ab式分类和mlp分类头式分类
"""
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
from utils.register import Register

prompt_template = """You are given an image and a question. 
Image: {image}
Question: {question}
if you can get the answer from the image, please input 'A', otherwise, please input 'B'.
Answer:"""

@Register(name="gradio_classify_internvl2_preprocessor")
class GradioClassifyInternVL2Preprocessor(BasePreprocessor):
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
        self.train_transform = build_transform(
            is_train=True, input_size=448, pad2square=pad2square
        )
        self.test_transform = build_transform(
            is_train=False, input_size=448, pad2square=pad2square
        )

    def get_prompt(self, label, **kwargs):
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

        question = item["question"]
        image_path = item["image_path"]
        
        # pixel_values, num_tiles = self.get_pixel_values(
        #     image_path, self.train_transform
        # )
        test_pixel_values, num_tiles = self.get_pixel_values(image_path, self.test_transform)
        _, test_conversation = self.get_prompt(
            label="fake label", image="<image>", question=question
        )
        # train_inputs = preprocess_internlm(
        #     template_name=self.template_name,
        #     sources=[train_conversation],
        #     tokenizer=self.tokenizer,
        #     num_image_token_list=[
        #         self.num_image_token * num_tile for num_tile in num_tiles
        #     ],
        #     text_only=False,
        #     ds_name="train_mpdocvqa",
        #     system_message=self.system_message,
        #     num_image=1,
        # )
        test_inputs = preprocess_internlm_for_test(
            template_name=self.template_name,
            sources=[test_conversation],
            tokenizer=self.tokenizer,
            num_image_token_list=[
                self.num_image_token * num_tile for num_tile in num_tiles
            ],
            text_only=False,
            ds_name="test_mpdocvqa",
            system_message=self.system_message,
            num_image=1,
        )
        model_inputs = dict()
        extra = dict(
            image_path=image_path,
            question=question,
            test_conversation=test_conversation,
        )

        model_inputs.update(
            dict(
                extra = extra,
                # pixel_values=pixel_values,
                test_pixel_values=test_pixel_values,
                # input_ids=train_inputs["input_ids"].squeeze(),
                # attention_mask=train_inputs["attention_mask"].squeeze(),
                # labels = train_inputs["labels"].squeeze(),
                image_flags=torch.tensor([1] * test_pixel_values.size(0), dtype=torch.long),
                # cls_label=torch.tensor(cls_label, dtype=torch.long),
                test_input_ids=test_inputs["input_ids"],
                test_attention_mask=test_inputs["attention_mask"],
                test_conversation=test_conversation,
            )
        )
        return model_inputs
