import os
from typing import Any, List, Tuple
import json
from tester.default_tester import DefaultTester
from dataset.qwen2vl.template import Qwen2VLTemplate
from copy import deepcopy

from transformers import Qwen2VLProcessor, Qwen2Tokenizer

from logger import logger

from utils.register import Register

"""
    进行固定的k轮推理对话
    传入的数据对多轮的形式，但是机器的输出不进行采用，每经过一轮之后，
    填入机器的输入，然后进行下一轮的对话
    first turn:
        first prompt: xxx
        assistant: model response
        second prompt: xxx
        assistant: model response
        [
            '如果'
            {'role': 'user', 'content': [{'type': 'text', 'text': 'aaa'}, {'type': 'image', 'image': 'image_path'}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': 'unknown'}]}
            {'role': 'user', 'content': []} ...
        ]

"""


@Register(name="qwen2vl_multi_conversation_tester")
class Qwen2VLMultiConversationTester(DefaultTester):
    def __init__(
        self,
        model_config,
        test_dataset,
        output_dir,
        # 自己填写
        model_path,
        dataloader_config,
        conversation_key: str = "test_conversation",
        system_message: str = "You are a helpful assistant.",
        test_raw_model: bool = False,
        metrics=[],
        max_steps: int = -1,
        checkpoint_list: List[str] = [],
        # qwen2vl specific
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    ):
        super().__init__(
            model_config,
            test_dataset,
            output_dir,
            dataloader_config,
            test_raw_model,
            metrics,
            max_steps,
            checkpoint_list,
        )
        self.processor = Qwen2VLProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)
        self.template = Qwen2VLTemplate()
        self.template.set_system_message(system_message)
        self.conersation_key = conversation_key

    def run_model(self, model, batch_input: dict) -> Tuple[float, List[Any]]:
        """
        List[Any], model output,一个batch的输出
        """

        test_conversation = batch_input["test_conversation"]
        candidate_answers = batch_input["candidate_answers"]
        qid = batch_input["extra"][0]["qid"]
        assert (
            len(test_conversation) == 1
        ), "MultiTurn Conversation Only Support batch_size = 1"

        test_conversation = test_conversation[0]
        candidate_answers = candidate_answers[0]
        assert len(test_conversation) % 2 == 0, "conversation should be even"

        history = []
        for i in range(0, len(test_conversation), 2):

            user_prompt = test_conversation[i]
            # assistant_response = test_conversation[i + 1]

            resp, history, seq_len = self.chat(
                history, user_prompt, model, return_seq_len=True
            )
            logger.info(
                f"qid {qid}, turn {i//2},seq_len:{seq_len} => {json.dumps(history[-1],ensure_ascii=False)}"
            )
            # logger.info(f"{json.dumps(user_prompt,ensure_ascii=False)} => {json.dumps(history[-1],ensure_ascii=False)}")
        if "My final answer is" not in resp:
            # 无法得到最终答案，直接取第一个候选答案
            logger.warning(f"qid: {qid} => has no final answer")
            pred_answer = candidate_answers[0]
        else:
            pred_answer = resp.split("My final answer is")[-1].strip()
        return None, [{"history": history, "pred_answer": pred_answer}]

    def run_model_on_batch(self, model, batch_input: dict) -> Tuple[float, List[Any]]:
        """
        舍弃函数，请不要使用
        List[Any], model output,一个batch的输出
        """
        test_conversation = batch_input["test_conversation"]
        candidate_answers = batch_input["candidate_answers"]
        bsz = len(test_conversation)
        # 初始化history
        conv_turn = len(test_conversation[0])
        history_batch = []
        for _ in range(bsz):
            history_batch.append([])
        for i in range(0, conv_turn, 2):
            # 单独进行对话预处理
            prompt_batch = []
            image_batch = []
            for j in range(bsz):
                user_prompt = test_conversation[j][i]
                prompt, image_list, his = self.preprocess(history_batch[j], user_prompt)
                history_batch[j] = his
                prompt_batch.append(prompt)
                image_batch.extend(image_list)
            # 进行batch_chat
            resp_batch = self.batch_chat(prompt_batch, image_batch, model)
            # 推理结束之后将内容添加到history中
            for j in range(bsz):
                resp = resp_batch[j]
                history_batch[j].append(
                    {"role": "assistant", "content": [{"type": "text", "text": resp}]}
                )
            # 进行下一轮对话
        # 对话轮数结束，统计答案
        pred_answer_batch = []
        for i, resp in enumerate(resp_batch):

            if "My final answer is" not in resp:
                qid = batch_input["extra"][i]["qid"]
                logger.warning(f"qid: {qid}, => has no final answer")
                # 无法得到最终答案，直接取第一个候选答案
                pred_answer_batch.append(candidate_answers[i][0])
            else:
                pred_answer_batch.append(resp.split("My final answer is")[-1].strip())

        result = []
        for his, pred_answer in zip(history_batch, pred_answer_batch):
            result.append({"history": his, "pred_answer": pred_answer})
        return None, result

    def get_prompt(self, messages):
        image_list = []
        self.template.clear_messages()
        for message in messages:

            role = message["role"]
            msg, imgs = self.template.parse_message_by_openai_with_imageplaceholder(
                message["content"]
            )
            image_list.extend(imgs)
            self.template.add_message(role, msg)
        prompt = self.template.get_prompt(add_generation_prompt=True)
        return prompt, image_list

    def chat(self, history: list, query, model, return_seq_len: bool = False):
        """
        history = [
            {"role": 'user', 'content': [{'type': 'text', 'text': 'aaa'}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': 'bbb'}]}
        ]
        query = {'role': 'user', 'content': [{'type': 'text', 'text': 'ccc'}]}
        Returns:
            resp = {"role": 'assistant', 'content': [{'type': 'text', 'text': 'ddd'}]}
            new_history
        """

        assert len(history) % 2 == 0, "history should be even"
        assert query["role"] == "user", "query role should be user"
        history = deepcopy(history)
        history.append(query)
        # 预处理
        prompt, image_list = self.get_prompt(history)
        model_inputs = self.processor(
            text=[prompt],
            images=image_list if len(image_list) > 0 else None,
            videos=None,
            # padding="max_length",
            # max_length=self.max_seq_length,
            return_tensors="pt",
        )
        seq_len = model_inputs["input_ids"].shape[1]
        resp = model.batch_chat(
            pixel_values=model_inputs["pixel_values"],
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            image_grid_thw=model_inputs["image_grid_thw"],
            video_grid_thw=None,
            return_generation_ids=False,
        )
        resp = resp[0]

        history.append(
            {"role": "assistant", "content": [{"type": "text", "text": resp}]}
        )
        ret = (resp, history) if not return_seq_len else (resp, history, seq_len)
        return ret

    # def preprocess(self, history:list, query):
    #     """
    #         进行单个数据的输出预处理
    #         得到含有prompt和image_list的数据
    #     """

    #     assert len(history) % 2 == 0, "history should be even"
    #     assert query["role"] == "user", "query role should be user"
    #     history = deepcopy(history)
    #     history.append(query)
    #     # 预处理
    #     prompt, image_list = self.get_prompt(history)
    #     return prompt, image_list, history

    # def batch_chat(self, prompt_list, image_list, model):
    #     """
    #         进行batch_chat,batch内的对话均推进一轮
    #     """
    #     model_inputs = self.processor(
    #         text=prompt_list,
    #         images=image_list if len(image_list) > 0 else None,
    #         videos=None,
    #         padding="max_length",
    #         max_length=3072,
    #         return_tensors="pt",
    #     )
    #     resp = model.batch_chat(
    #         pixel_values=model_inputs["pixel_values"],
    #         input_ids=model_inputs["input_ids"],
    #         attention_mask=model_inputs["attention_mask"],
    #         image_grid_thw=model_inputs["image_grid_thw"],
    #         video_grid_thw=None,
    #         return_generation_ids=False,
    #         max_new_tokens=512,
    #     )
    #     return resp
