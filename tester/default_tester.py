from typing import Any, Dict, List, Tuple
import json
import os
from tqdm import tqdm
import gc
import itertools

import torch
from torch.utils.data import DataLoader

from trainer.trainer_utils import load_model
from metrics.build import build_metrics
from utils.utils import (
    get_cls_or_func,
    load_state_dict_from_ckpt,
    check_model_state_dict_load,
    delete_not_used_key_from_batch,
)
from utils.dist_variable import DistVarible
from utils.register import Register
from dataset.default_collator import default_collate
from logger import logger


@Register(name="default_tester")
class DefaultTester:
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
    ):

        self.model_config = model_config
        self.metrics_config = metrics
        self.test_dataset = test_dataset
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self._create_dataloader(dataloader_config)
        self.checkpoint_list = checkpoint_list
        self.checkpoint_0_path = "/checkpoint-0/pytorch_model.bin"
        if test_raw_model:
            self.checkpoint_list.insert(0, self.checkpoint_0_path)

    def test(self):
        for checkpoint_path in self.checkpoint_list:
            model = self.load_checkpoint(checkpoint_path)
            # model to cuda
            if torch.cuda.is_available():
                model = model.to(f"cuda:{DistVarible.local_rank}")
            result = self.test_one_checkpoint(model)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            # save result to output_dir
            save_path = os.path.join(
                self.output_dir, f"{checkpoint_path.split('/')[-2]}-result.json"
            )
            self.save_result(result, save_path)
            # cal metrics
            self.cal_metrics(
                save_path, save_path.replace("result.json", "metrics.json")
            )

    def run_model(self, model, batch: dict) -> Tuple[float, List[Any]]:
        """
            return float, loss
            List[Any], model output
            allow type: 1. float, str, int, torch.Tensor, the will be saved directly
                        2. dict, key-value pair will be extracted and saved
        """
        model_input_batch, delete_batch = delete_not_used_key_from_batch(model, batch)
        outputs = model(**model_input_batch)
        batch.update(delete_batch)
        return outputs

    def cal_metrics(self, result_path, save_path):
        metrics_result = {
            "result_path": result_path,
        }
        for m in self.metrics_config:
            metrics = build_metrics(m, result_path=result_path)
            # metrics_type = m.get("type", None)
            metrics_modules = metrics.__class__.__module__.split(".")
            metrics_modules.append(metrics.__class__.__name__)
            metrics_name = ".".join(metrics_modules)
            metrics_value, metrics_details = metrics.compute_metrics()
            logger.info(
                f"{result_path}=> {metrics_name}: {metrics_value}, metrics_details: {metrics_details}"
            )
            m
            
            metrics_result[metrics_name] = metrics_value
            metrics_result[f"{metrics_name}_details"] = metrics_details

        with open(save_path, "w") as f:
            json.dump(metrics_result, f, ensure_ascii=False, indent=2)

    def save_result(self, result, save_path):
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    @torch.no_grad()
    def test_one_checkpoint(self, model) -> List[Dict[str, Any]]:
        result = []
        for i, batch in enumerate(
            tqdm(self.dataloader, desc="Testing", total=self.max_steps)
        ):
            if i >= self.max_steps:
                break
            _, output = self.run_model(
                model, batch
            )  # loss, score[torch.tensor, list, or list of dict]
            if isinstance(output, torch.Tensor):
                output = output.detach().to(torch.float32).cpu().numpy().tolist()
            if isinstance(output, list):
                result.extend(self._split_output_and_save_extra_info(output, batch))
            else:
                raise ValueError("output should be list or tensor")
        return result

    def _split_output_and_save_extra_info(
        self, model_outputs: list, batch: Dict[str, Any]
    ) -> List[Dict[str, Any]]:

        save_keys = getattr(self.test_dataset.preprocessor, "save_keys",[])
        has_extra = 'extra' in batch
        if len(save_keys) == 0 and not has_extra:
            logger.warning(
                f"save_keys = {save_keys} has_extra: {has_extra}, please check if you have set save_keys in preprocessor"
            )
        if has_extra and len(save_keys) > 0:
            logger.warning(
                f"save_keys = {save_keys} has_extra: {has_extra}, normally you should only set one of them"
            )
        result = []
        batch_keys = list(batch.keys())
        for i, output in enumerate(model_outputs):
            one_data = dict()
            if isinstance(output, dict):
                one_data.update(output)
            else:
                one_data["model_output"] = output
            for key in batch_keys:
                if key in save_keys:
                    one_data[key] = batch[key][i]
                if key == 'extra' and isinstance(batch[key][i], dict):
                    one_data.update(batch[key][i])
            result.append(one_data)
        return result

    def _create_dataloader(self, dataloader_config: dict):
        # handle collator
        max_steps = dataloader_config.pop("max_steps", -1)

        data_collator = dataloader_config.pop("data_collator", None)
        data_collator = (
            get_cls_or_func(data_collator) if data_collator else default_collate
        )
        batch_size = dataloader_config.get("batch_size", 1)
        self.dataloader = DataLoader(
            self.test_dataset,
            collate_fn=data_collator,
            drop_last=False,
            **dataloader_config,
        )
        self.max_steps = max_steps
        if max_steps == -1:
            self.max_steps = len(self.dataloader)

    def load_checkpoint(self, checkpoint_path, state_dict_map_location="cpu"):
        gc.collect()
        torch.cuda.empty_cache()
        
        model = load_model(self.model_config)
        if checkpoint_path != self.checkpoint_0_path:
            state_dict = load_state_dict_from_ckpt(checkpoint_path, state_dict_map_location)
            check_model_state_dict_load(model, state_dict)
            model.load_state_dict(state_dict, strict=True)
            del state_dict
            gc.collect()
            torch.cuda.empty_cache()
            logger.info(
                f"[Model load] Model checkpoint loaded successfully from {checkpoint_path}."
            )
        else:
            logger.info(
                f"[Model load], load model without checkpoint, only load model config"
            )
        model.eval()
        model.requires_grad_(False)
        return model


if __name__ == "__main__":
    tester = DefaultTester(
        model_config={},
        test_dataset={},
        output_dir="output",
        dataloader_config={},
        metrics=[
            {"type": "metrics.auc_metrics.AUCMetrics"},
            {
                "type": "metrics.docvqa_metrics.mpdocvqa_page_accuracy_metrics.MPDocVQPageAccuracyMetrics"
            },
        ],
        checkpoint_list=[],
    )
    tester.cal_metrics(
        "/home/klwang/code/elaina/outputs/MPDocVQA/trainer_test_output/test_result/checkpoint-35000-result.json",
        "./metrics.json",
    )
