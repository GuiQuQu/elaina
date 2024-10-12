import os
from typing import Any, List, Tuple
import json
from tester.default_tester import DefaultTester

from models.docvqa.internvl2.tokenization_internlm2 import InternLM2Tokenizer

def open_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

class InternVL2Tester(DefaultTester):
    def __init__(
        self,
        model_config,
        test_dataset,
        output_dir,
        # 自己填写
        model_path,
        dataloader_config,
        generation_config,
        metrics,
        max_steps: int = -1,
        checkpoint_list: List[str] = [],
    ):
        super().__init__(
            model_config,
            test_dataset,
            output_dir,
            dataloader_config,
            metrics,
            max_steps,
            checkpoint_list,
        )
        self.tokenizer = InternLM2Tokenizer.from_pretrained(model_path)
        self.generation_config = generation_config

    def run_model(self, model, batch: dict) -> Tuple[float, List[Any]]:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        test_conversation = batch["test_conversation"]  # list of dict
        questions = [t[0]["value"] for t in test_conversation]
        test_pixel_values = batch["test_pixel_values"]
        test_pixel_values = test_pixel_values.to(device=device, dtype=dtype)
        reponses = model.batch_chat(
            tokenizer=self.tokenizer,
            pixel_values=test_pixel_values,
            questions=questions,
            generation_config=self.generation_config,
            num_patches_list = batch['num_tiles'],
        )
        return None,reponses
