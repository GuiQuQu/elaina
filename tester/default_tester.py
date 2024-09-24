import json
import os
from typing import Any, Dict, List
import torch
import torch.utils
from torch.utils.data import DataLoader
import gc

import torch.utils.data

from trainer.trainer_utils import load_model
from utils.utils import (
    get_cls_or_func,
    load_state_dict_from_ckpt,
    check_model_state_dict_load,
    delete_not_used_key_from_batch,
)
from utils.dist_variable import _DistVarible
from dataset.default_collator import default_collator


class DefaultTester:
    def __init__(
        self,
        model_config,
        test_dataset,
        output_dir,
        dataloader_config,
        checkpoint_list=[],
    ):

        self.model_config = model_config
        self.test_dataset = test_dataset
        self.output_dir = output_dir

        self.dataloader = self._create_dataloader(dataloader_config)
        self.checkpoint_list = checkpoint_list

    def test(self):
        for checkpoint_path in self.checkpoint_list:
            model = self.load_checkpoint(checkpoint_path)
            # model to cuda
            if torch.cuda.is_available():
                model = model.to(f"cuda:{_DistVarible.local_rank}")
            result = self.test_one_checkpoint(model)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            # save result to output_dir
            save_path = os.path.join(
                self.output_dir, f"{checkpoint_path.split('/')[-2]}-result.json"
            )
            self.save_result(result, save_path)
        
        # TODO metrics

    def run_model(model, batch):
        model_input_batch = delete_not_used_key_from_batch(model, batch)
        return model(**model_input_batch)

    def save_result(self, result, save_path):
        with open(save_path, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    @torch.no_grad()
    def test_one_checkpoint(self, model) -> List[Dict[str, Any]]:
        result = []

        for _, batch in enumerate(self.dataloader):
            _, output = self.run_model(model, batch)  # socre, score
            if isinstance(output, torch.tensor):
                output = output.detach().to(torch.float32).cpu().numpy().tolist()
                result.extend(self._split_output_and_batch(output, batch))
        return result

    def _split_output_and_batch(
        model_outputs: list, batch: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        model_output : List[output]
        batch : Dict[str, Any]
        """
        result = []
        batch_keys = list(batch.keys())
        for i, output in enumerate(model_outputs):
            one_data = dict(model_output=output)
            for key in batch_keys:
                one_data[key] = batch[key][i]
            result.append(one_data)
        return result

    def _create_dataloader(self, dataloader_config):
        # handle collator
        if "data_collator" in self.dataloader:
            data_collator = get_cls_or_func(self.dataloader.pop("data_collator")) 
        else:
            data_collator = default_collator
        self.dataloader = DataLoader(
            self.test_dataset, collate_fn=data_collator, **dataloader_config
        )

    def load_checkpoint(self, checkpoint_path):
        gc.collect()
        torch.cuda.empty_cache()
        model = load_model(self.model_config)
        state_dict = load_state_dict_from_ckpt(checkpoint_path)
        check_model_state_dict_load(model, state_dict)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.requires_grad_(False)
        return model
