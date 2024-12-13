"""
    ensemable 多个llm预测得到的答案，然后基于qwen2vl-2b，不经过微调，让模型
    进行选择，这是一种简单的投票策略，看是否可以得到模型效果的提升。
"""

from collections import defaultdict
import random
from typing import List
from PIL import Image
import torch
import json

from transformers import Qwen2VLProcessor
from dataset.qwen2vl.template import Qwen2VLTemplate
from dataset.base_preprocessor import BasePreprocessor
from dataset.qwen2vl.preprocess import generate_labels
from logger import logger
from utils.register import Register

from dataset.docvqa.ocr2layout.sp_ocr2layout import transform_ocr2layout
from dataset.docvqa.docvqa_utils import truncate_layout

# prompt_template = """You are given an image and a question. 
# Image: {image}
# Question: {question}
# Please answer the question based on the image.
# You should extract the answer from the text in the image without changing the order and form of the words.
# Answer: """

# prompt_template_add_layout = """You are given an image,its corresponding string layout and a question.
# Image: {image}
# String Layout: 
# {layout}
# Question: {question}
# Please answer the question based on the image and its its corresponding string layout.
# You should extract the answer from the text in the image without changing the order and form of the words.
# Answer:"""

# 支持指定数量的agent
# 这个prompt只看实验的结果来说效果很烂，
# 在该多选的地方不多选，在该根据自己判断的地方无法直接判断
prompt_agent_template = """You are given an image and a question.
Image: {image}
Question: {question}
Now there are {agent_cnt} agents, each of them will give the answer about the question based on the image.
Please choose the correct answer from their answers.
Generally, the correct answer is the one that most agents choose.
If you think no agent gives the correct answer, you can answer the question by yourself.
"""

# prompt_example
prompt_example = """You are given an image and a question.
Image: {image}
Question: {question}
Now there are {agent_cnt} agents, each of them will give the answer about the question based on the image.
Please choose the correct answer from their answers.
If you think no agent gives the correct answer, you can answer the question by yourself.
Agent 1 Answer: {agent_1_answer}
Agent 2 Answer: {agent_2_answer}
Agent 3 Answer: {agent_3_answer}
Your answer:
"""



def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def format_the_complete_agent_template(
    unfinish_template: str,
    image,
    question,
    agents_answers: List[str],
    the_last_part: str = "Your answer:",
):
    """
    根据给定的模板，和传入的angent的答案的数量，填充带有agent答案的模板
    """
    agent_cnt = len(agents_answers)
    finish_prompt = unfinish_template.format(
        image=image, question=question, agent_cnt=agent_cnt
    )
    for i, agent_answer in enumerate(agents_answers):
        finish_prompt += f"Agent {i+1} Answer: {agent_answer}\n"
    if len(the_last_part) > 0:
        finish_prompt += the_last_part
    return finish_prompt


@Register(name="docvqa_ensemable_qwen2vl_preprocessor")
class DocVQAEnsemableQwen2VLPreprocessor(BasePreprocessor):
    def __init__(
        self,
        model_path,
        # use_ocr = False,
        few_shot: int = 0,
        agents_answer_paths=[],
        max_layout_length=1024,
        max_seq_length=1024,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        system_message="You are a helpful assistant.",
    ) -> None:
        super().__init__()

        self.template = Qwen2VLTemplate()
        self.max_seq_length = max_seq_length
        self.system_message = system_message

        #
        # self.use_ocr = use_ocr
        self.max_layout_length = max_layout_length
        # self.prompt_template = prompt_template_add_layout if use_ocr else prompt_template
        self.prompt_template = prompt_agent_template
        self.few_shot_num = few_shot

        self.agents_answers_list = self.load_agents_answer(agents_answer_paths)
        # self.tokenizer: Qwen2Tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        # self.tokenizer.model_max_length = max_seq_length
        self.processor = Qwen2VLProcessor.from_pretrained(
            model_path, min_pixels=min_pixels, max_pixels=max_pixels
        )

    def get_prompt(self, prompt_template, answer: str, **kwargs):
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

    def prompt_str2_opanai_format(self, prompt_str, assistant_answer):
        train_ret = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_str}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_answer}],
            },
        ]
        test_ret = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_str}],
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
        layout = transform_ocr2layout(ocr_path, placeholder=" ")
        layout, is_truncated = truncate_layout(
            layout,
            tokenizer=self.processor.tokenizer,
            max_token_length=self.max_layout_length,
        )
        return layout

    def load_agents_answer(self, angents_answer_paths):
        agents_answers: list[dict] = []
        for path in angents_answer_paths:
            agent_result = load_json(path)
            # 整理为dict形式
            save_elem = defaultdict(dict)
            for item in agent_result:
                qid = item["qid"]
                pred_answer: str = item["model_output"]
                true_answers: List[str] = item["answers"]
                _from = path.split("/")[2]
                save_elem[qid] = dict(
                    qid=qid, pred_answer=pred_answer, true_answers=true_answers, 
                )
                save_elem[qid]['from'] = _from
            # append to the final list
            agents_answers.append(save_elem)
        return agents_answers

    def preprocess(self, item):

        qid = item["qid"]
        question = item["question"]
        image_path = item["image_path"]
        ocr_path = item["ocr_path"]
        answers = item["answers"]
        image = Image.open(image_path).convert("RGB")

        # layout = self.transform_ocr2layout(ocr_path)
        agents_answers = [
            (agent[qid]["pred_answer"], agent[qid]['from']) for agent in self.agents_answers_list
        ]
        prompt_text = format_the_complete_agent_template(
            self.prompt_template,
            image=self.template.image_placeholder,
            question=question,
            agents_answers=[agent[0] for agent in agents_answers],
            the_last_part="Your answer:",
        )

        train_conversation, test_conversation = self.prompt_str2_opanai_format(
            prompt_str=prompt_text,
            assistant_answer=random.choice(answers),
        )
        train_text = self.get_text(train_conversation, add_generation_prompt=False)
        test_text = self.get_text(test_conversation, add_generation_prompt=True)

        # train_inputs = self.processor(
        #     text = [train_text],
        #     images = [image],
        #     videos=None,
        #     padding="max_length",
        #     max_length = self.max_seq_length,
        #     return_tensors="pt",
        # )
        # if train_inputs['input_ids'].size(-1) != self.max_seq_length:
        #     logger.warning(f"input_ids size: {train_inputs['input_ids'].size()} is not equal to max_seq_length: {self.max_seq_length}")
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
            image_path=image_path,
            ocr_path=ocr_path,
            answers=answers,
            agents_answers=agents_answers,
            test_conversation=test_conversation,
        )
        # model_inputs.update(extra)
        # self.save_keys = list(extra.keys())
        model_inputs.update(
            dict(
                extra=extra,
                # pixel_values=train_inputs["pixel_values"],
                # image_grid_thw=train_inputs["image_grid_thw"].squeeze(),
                # input_ids=train_inputs["input_ids"].squeeze(),
                # attention_mask=train_inputs["attention_mask"].squeeze(),
                # labels=train_labels.squeeze(),
                # test input
                test_pixel_values=test_inputs["pixel_values"],
                test_image_grid_thw=test_inputs["image_grid_thw"].squeeze(),
                test_input_ids=test_inputs["input_ids"].squeeze(),
                test_attention_mask=test_inputs["attention_mask"].squeeze(),
            )
        )
        return model_inputs
