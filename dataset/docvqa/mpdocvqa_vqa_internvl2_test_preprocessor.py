"""
    internvl2,测试的，需要读取分类结果文件, 然后提供选择的图像 preprocessor
"""
from collections import defaultdict
import random
from PIL import Image
import torch
import json

from models.docvqa.internvl2.tokenization_internlm2 import InternLM2Tokenizer
from dataset.docvqa.preprocess import (
    build_transform,
    preprocess_internlm,
    preprocess_internlm_for_test,
    dynamic_preprocess,
)
from dataset.base_preprocessor import BasePreprocessor
from logger import logger

prompt_template = """You are given an image and a question. 
Image: {image}
Question: {question}
Please answer the question based on the image.
You should extract the answer from the text in the image without changing the order and form of the words.
Answer: """

def internvl2_concat_collator(batch):
    assert isinstance(batch, list)

    elem = batch[0]
    assert isinstance(elem, dict), f"elem type: {type(elem)}, expected type: dict"
    ret_batch = {}
    keys = list(elem.keys())
    for k in keys:
        if isinstance(elem[k], torch.Tensor):
            shape = elem[k].shape
            if all([d[k].shape == shape for d in batch]) and k not in [
                "pixel_values",
                "test_pixel_values",
                "image_flags",
            ]:
                ret_batch[k] = torch.stack([d[k] for d in batch], dim=0)
            else:
                ret_batch[k] = torch.cat([d[k] for d in batch], dim=0)
        else:
            ret_batch[k] = [d[k] for d in batch]
    return ret_batch


class MPDocVQAVQAInternVL2TestPreprocessor(BasePreprocessor):
    def __init__(
        self,
        model_path,
        classify_result_path:str,
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
        # sft训练不考虑做图像增强变换
        self.train_transform = build_transform(
            is_train=False, input_size=448, pad2square=pad2square
        )
        self.test_transform = build_transform(
            is_train=False, input_size=448, pad2square=pad2square
        )
        self.qid2classifyitems = self.groupby_classify_result(classify_result_path)

    def groupby_classify_result(self, classify_result_path):
        qid2items = defaultdict(dict)
        with open(classify_result_path, 'r', encoding='utf-8') as f:
            classify_result = json.load(f)
        for item in classify_result:
            qid = item['qid']
            page_id = item['image_path'].split('/')[-1].split('.')[0]
            qid2items[qid][page_id] = item['model_output']
        return qid2items


    def get_prompt(self, answer, **kwargs):
        prompt = prompt_template.format(**kwargs)
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

    def preprocess(self, item):
        qid = item["qid"]
        question = item["question"]
        documents = item["documents"]
        true_answer_page_idx = item["true_answer_page_idx"]
        answers = item["answers"]
        classify_items:dict = self.qid2classifyitems[qid]
        # 排序根据score排序所有的page_id,选择返回分数最高的page_id
        # add score to documents
        for i, doc in enumerate(documents):
            page_id = doc['page_id']
            doc['is_true_page'] = i == true_answer_page_idx
            if page_id in classify_items:
                doc['score'] = classify_items[page_id]
            else:
                raise ValueError(f"page_id: {page_id} not found in classify_result")
        documents = sorted(documents, key=lambda x: x['score'], reverse=True)
        top1_image_path = documents[0]['image_path']
        top1_ocr_path = documents[0]['ocr_path']
        test_pixel_values, num_tiles = self.get_pixel_values(top1_image_path, self.test_transform)
        train_conversation, test_conversation = self.get_prompt(
            answer=random.choice(answers), image="<image>", question=question
        )
        model_inputs = dict()
        extra = dict(
            qid=qid,
            answers=answers,
            documents=documents,
            
        )
        model_inputs.update(extra)
        self.save_keys = ['qid', 'answers', 'documents','test_conversation']

        model_inputs.update(
            dict(
                # pixel_values=pixel_values,
                # input_ids=train_inputs["input_ids"].squeeze(),
                # attention_mask=train_inputs["attention_mask"].squeeze(),
                # labels=train_inputs["labels"].squeeze(),
                # image_flags=torch.tensor([1] * pixel_values.size(0), dtype=torch.long),
                test_conversation=test_conversation,
                test_pixel_values=test_pixel_values,
                num_tiles=num_tiles[0], # 单图，因此只选择第一个图像被划分出来的patch数量
            )
        )
        
        return model_inputs
