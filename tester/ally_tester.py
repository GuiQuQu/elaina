import os
import json
from typing import List, Tuple, Any

from tester.default_tester import DefaultTester

from models.docvqa.internvl2.tokenization_internlm2 import InternLM2Tokenizer


def open_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


class InternVL2AllyTester(DefaultTester):
    def __init__(
        self,
        model_config,
        test_dataset,
        output_dir,
        model_path,
        dataloader_config,
        generation_config,
        metrics=[],
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
        self.generate_config = generation_config

    def run_model(self, model, batch: dict) -> Tuple[float, List[Any]]:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        classify_input_ids = batch["classify_input_ids"]
        classify_attention_mask = batch["classify_attention_mask"]
        pixel_values = batch["pixel_values"]
        pixel_values = pixel_values.to(device=device, dtype=dtype)
        classify_input_ids = classify_input_ids.to(device=device)
        classify_attention_mask = classify_attention_mask.to(device=device)

        _, classify_outputs = model.classify_forward(
            pixel_values=pixel_values,
            classify_input_ids=classify_input_ids,
            classify_attention_mask=classify_attention_mask,
            image_flags=batch["image_flags"],
            cls_label=None,
        )

        test_conversation = batch["vqa_train_conversation"]  # list of dict
        questions = [t[0]["value"] for t in test_conversation]
        responses: List[str] = model.batch_chat(
            tokenizer=self.tokenizer,
            pixel_values=pixel_values,
            questions=questions,
            generation_config=self.generate_config,
            num_patches_list=batch["num_tiles"],
        )
        assert len(responses) == len(classify_outputs)
        result = [{'score' : score, 'resp' : resp} for score, resp in zip(classify_outputs, responses)]

        return None, result
