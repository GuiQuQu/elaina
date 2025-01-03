from typing import List
from transformers import Qwen2VLProcessor, Qwen2Tokenizer
from PIL import Image

from dataset.base_preprocessor import BasePreprocessor
from dataset.qwen2vl.template import Qwen2VLTemplate
from dataset.qwen2vl.preprocess import generate_labels
from logger import logger
from utils.register import Register



@Register(name="mpdocvqa_cot_mutli_conv_qwen2vl_preprocessor")
class MPDocVQACOTMultiConvQwen2VLPreprocessor(BasePreprocessor):
    def __init__(
        self,
        model_path,
        max_seq_length=1024,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        system_message="You are a helpful assistant.",
    ) -> None:
        super().__init__()
        pass
        self.template = Qwen2VLTemplate()
        self.max_seq_length = max_seq_length
        self.template.set_system_message(system_message)
        self.processor = Qwen2VLProcessor.from_pretrained(
            model_path, min_pixels=min_pixels, max_pixels=max_pixels
        )
    
    def preprocess(self, item):
        qid = item["qid"]
        question = item["question"]
        documents = item["documents"]
        true_answer_page_idx = item["true_answer_page_idx"]
        true_page = documents[true_answer_page_idx]
        true_image_path = true_page["image_path"]
        true_ocr_path = true_page["ocr_path"]
        answers = item["answers"]
        cot_conversation = item["cot_conversation"]

        image = Image.open(true_image_path).convert("RGB")
      
        # 生成对话
        self.template.clear_messages()
        for message in cot_conversation:
            role = message["role"]
            msg:str = message["content"][0]['text']
            # msg = msg.replace(f"<|{true_image_path}|>", self.template.image_placeholder,1)
            self.template.add_message(role, msg)
        
        prompt = self.template.get_prompt(add_generation_prompt=False)
        train_inputs = self.processor(
            text = [prompt],
            images = [image],
            videos=None,
            padding="max_length",
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        # 生成标签
        labels = generate_labels(
            train_inputs["input_ids"],
            [prompt],
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
            prompt = prompt,
            cot_conversation = cot_conversation,
        )

        model_inputs.update(
            dict(
                extra = extra,
                pixel_values=train_inputs["pixel_values"],
                image_grid_thw=train_inputs["image_grid_thw"].squeeze(),
                input_ids=train_inputs["input_ids"].squeeze(),
                attention_mask=train_inputs["attention_mask"].squeeze(),
                labels=labels.squeeze(),
            )
        )

        return model_inputs
