from typing import Any, List, Tuple
import json
from tester.default_tester import DefaultTester

from utils.register import Register
from tester.custom_tester import delete_not_used_key_from_batch_in_inference

from logger import logger


@Register(name="qwen2vl_cot_tester")
class Qwen2VLCoTTester(DefaultTester):
    def __init__(
        self,
        model_config,
        test_dataset,
        output_dir,
        dataloader_config,
        test_raw_model: bool = False,
        metrics=[],
        max_steps: int = -1,
        checkpoint_list: List[str] = [],
    ) -> None:
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

    def run_model(self, model, batch: dict) -> Tuple[float, List[Any]]:

        model_input_batch, delete_batch = delete_not_used_key_from_batch_in_inference(
            model, batch
        )

        _, outputs = model.inference_forward(**model_input_batch, max_new_tokens=1024)
        batch.update(delete_batch)

        ret = []
        for i, resp_with_cot in enumerate(outputs):
            try:
                t = resp_with_cot.split("```json")
                assert len(t) == 2
                t: str = t[1].split("```")[0].strip()
                prefix = "{'answer':"
                suffix = "}"
                answer = t[t.find(prefix) + len(prefix) : t.find(suffix)].strip()
            except Exception as e:
                qid = delete_batch["extra"][i]["qid"]
                logger.warning(
                    f"find answer error: qid={qid},e={e}, use default answer: 'fake_label'"
                )
                answer = "fake_label"
            ret.append(
                {
                    "pred_answer": answer,
                    "model_generated_response": resp_with_cot,
                }
            )
        return (None,)


@Register(name="qwen2vl_cot_testerv2")
class Qwen2VLCoTTesterV2(DefaultTester):
    def __init__(
        self,
        model_config,
        test_dataset,
        output_dir,
        dataloader_config,
        max_new_tokens: int = 1024,
        test_raw_model: bool = False,
        metrics=[],
        max_steps: int = -1,
        checkpoint_list: List[str] = [],
    ) -> None:
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
        self.max_new_tokens = max_new_tokens
    def run_model(self, model, batch: dict) -> Tuple[float, List[Any]]:

        model_input_batch, delete_batch = delete_not_used_key_from_batch_in_inference(
            model, batch
        )

        _, outputs = model.inference_forward(**model_input_batch, max_new_tokens=self.max_new_tokens)
        batch.update(delete_batch)
        breakpoint()
        ret = []
        for i, resp_with_cot in enumerate(outputs):
            answer = resp_with_cot.split("\n")[-1].strip()
            ret.append(
                {
                    "pred_answer": answer,
                    "model_generated_response": resp_with_cot,
                }
            )

        return None, ret
