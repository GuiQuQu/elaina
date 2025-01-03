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


@Register(name="mpdocvqa_cot_multi_conv_qwen2vl_test_preprocessor")
class MPDocVQACoTMultiConvQwen2VLTestPreprocessor(BasePreprocessor):
    def __init__(
        self,
        model_path,
        classify_result_path: str,
        max_seq_length=1024,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        reverse=True,
        system_message="You are a helpful assistant.",
    ) -> None:
        super().__init__()

        self.template = Qwen2VLTemplate()
        self.max_seq_length = max_seq_length
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

    def groupby_classify_result(self, classify_result_path):
        qid2items = defaultdict(dict)
        with open(classify_result_path, "r", encoding="utf-8") as f:
            classify_result = json.load(f)
        for item in classify_result:
            qid = item["qid"]
            page_id = item["image_path"].split("/")[-1].split(".")[0]
            qid2items[qid][page_id] = item["model_output"]
        return qid2items

    def preprocess(self, item):

        qid = item["qid"]
        question = item["question"]
        documents = item["documents"]
        true_answer_page_idx = item["true_answer_page_idx"]
        cot_conversation = item["cot_conversation"]
        # true_page = documents[true_answer_page_idx]
        # true_image_path = true_page["image_path"]
        # true_ocr_path = true_page["ocr_path"]
        answers = item["answers"]
        candidate_answers = [cand["pred_answer"] for cand in item["candidate_answers"]]

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

        # image = Image.open(top1_image_path).convert("RGB")

        # 为数据添加指定的检索top1图像
        for i, message in enumerate(cot_conversation):
            # role = message["role"]
            content = message["content"]
            has_image = False
            for c in content:
                if c["type"] == "text":
                    has_image = self.template.image_placeholder in c["text"]
            if has_image:
                cot_conversation[i]["content"].append(
                    {"type": "image", "image": f"file://{top1_image_path}"}
                )
                break

        model_inputs = dict()
        extra = dict(
            qid=qid,
            answers=answers,
            documents=documents,
        )
        model_inputs.update(
            dict(
                extra=extra,
                test_conversation=cot_conversation,
                candidate_answers=candidate_answers,
            )
        )
        return model_inputs
