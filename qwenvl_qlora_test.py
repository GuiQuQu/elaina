from typing import Dict, List, Any
import os
import json
import gc

import torch
from torch.utils.data import DataLoader
from peft import PeftModel, PeftConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig, GPTQConfig
from transformers import set_seed

# 内部函数
from models.docvqa.qwenvl.tokenization_qwen import QWenTokenizer
from dataset.build import build_dataset
from dataset.default_collator import default_collate
from metrics.build import build_metrics
from utils.utils import (
    load_config,
    get_cls_or_func,
    delete_not_used_key_from_batch
)
from utils.register import registry_pycls_by_path
from logger import logger


def load_qwenvl_lora(adapter_name_or_path):
    """
    加载微调之后的qwen-vl模型
    """
    peft_config = PeftConfig.from_pretrained(adapter_name_or_path, inference_mode=True)
    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    lora_model = PeftModel.from_pretrained(model, adapter_name_or_path)
    lora_model.eval()
    # gptq 量化的模型不能merge ...
    # lora_model.base_model.merge_and_unload()
    print(
        f"'{adapter_name_or_path}' loaded, dtype is '{next(lora_model.parameters()).dtype}'"
    )
    for _, p in model.named_parameters():
        p.requires_grad = False
    lora_model.base_model.use_cache = True
    return lora_model


def load_qwenvl_model(model_name_or_path: str, use_lora, q_lora):
    """
    load qwen-vl-chat-int4
    """
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    config.use_cache = True
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=(
            GPTQConfig(bits=4, disable_exllama=True) if use_lora and q_lora else None
        ),
    )
    return model

def get_stop_words_ids(chat_format, tokenizer):
    if chat_format == "raw":
        stop_words_ids = [tokenizer.encode("Human:"), [tokenizer.eod_id]]
    elif chat_format == "chatml":
        stop_words_ids = [[tokenizer.im_end_id], [tokenizer.im_start_id]]
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")
    return stop_words_ids

class QwenVLQLoraTester(object):
    def __init__(
        self,
        model_path,
        test_dataset,
        output_dir,
        dataloader_config,
        test_raw_model: bool = False,
        metrics=[],
        checkpoint_list: List[str] = [],
    ):
        self.model_path = model_path
        self.metrics_config = metrics
        self.test_dataset = test_dataset
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self._create_dataloader(dataloader_config)
        self.tokenizer:QWenTokenizer = QWenTokenizer.from_pretrained(model_path)
        self.checkpoint_list = checkpoint_list
        self.checkpoint_0_path = "parents/checkpoint-0"
        if test_raw_model:
            self.checkpoint_list.insert(0, self.checkpoint_0_path)

    def test(self):
        for checkpoint_path in self.checkpoint_list:
            model = self.load_checkpoint(checkpoint_path)
            # model to cuda
            if torch.cuda.is_available():
                LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
                model = model.to(f"cuda:{LOCAL_RANK}")
            result = self.test_one_checkpoint(model)
            del model
            gc.collect()
            torch.cuda.empty_cache
            # save result to output_dir
            save_path = os.path.join(
                self.output_dir, f"{checkpoint_path.split('/')[-1]}-result.json"
            )
            self.save_result(result, save_path)
            # cal metrics
            self.cal_metrics(
                save_path, save_path.replace("result.json", "metrics.json")
            )

    def save_result(self, result, save_path):
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    def load_checkpoint(self, checkpoint_path):
        gc.collect()
        torch.cuda.empty_cache()
        if checkpoint_path == self.checkpoint_0_path:
            model = load_qwenvl_model(self.model_path, use_lora=True, q_lora=True)
            logger.info(
                f"[Model load], load qwenvl without checkpoint, only load model from {self.model_path}"
            )
        else:
            model = load_qwenvl_lora(checkpoint_path)
            logger.info(
                f"[Model load] Model checkpoint loaded successfully from {checkpoint_path}."
            )
        model.eval()
        model.requires_grad_(False)
        return model

    def run_model(self, model, batch: dict) -> List[Any]:
        # model_input_batch, delete_batch = delete_not_used_key_from_batch(model, batch)
        test_input_ids = batch["test_input_ids"]
        test_attention_mask = batch["test_attention_mask"]
        generation_config = model.generation_config
        stop_words_ids = get_stop_words_ids(
            generation_config.chat_format, self.tokenizer
        )

        generated_ids = self.generate(
            input_ids=test_input_ids,
            attention_mask=test_attention_mask,
            stop_words_ids=stop_words_ids,
            return_dict_in_generate=False,
            generation_config=generation_config,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(test_input_ids, generated_ids)
        ]
        output_text = self.tokenizer.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text
    
    def _split_output_and_save_extra_info(
        self, model_outputs: list, batch: Dict[str, Any]
    ) -> List[Dict[str, Any]]:

        save_keys = getattr(self.test_dataset.preprocessor, "save_keys", [])
        has_extra = "extra" in batch
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
                if key == "extra" and isinstance(batch[key][i], dict):
                    one_data.update(batch[key][i])
            result.append(one_data)
        return result

    @torch.no_grad()
    def test_one_checkpoint(self, model) -> List[Dict[str, Any]]:
        result = []
        for i, batch in enumerate(
            tqdm(self.dataloader, desc="Testing", total=self.max_steps)
        ):
            if i >= self.max_steps:
                break
            output = self.run_model(model, batch)  # List[Any] or torch.Tensor
            if isinstance(output, torch.Tensor):
                output = output.detach().to(torch.float32).cpu().numpy().tolist()
            if isinstance(output, list):
                result.extend(self._split_output_and_save_extra_info(output, batch))
            else:
                raise ValueError("output should be list or tensor")
        return result

    def _create_dataloader(self, dataloader_config):
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


def initialize_elaina_by_config(config):
    registry_paths = config["registry_paths"]
    for path in registry_paths:
        registry_pycls_by_path(path)

    set_seed(config["seed"])


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def get_tester_config(elaina_config):
    tester_config = elaina_config["tester"]
    tester_config.pop("type", None)
    # add output_dir
    # add output_dir
    output_dir = elaina_config.get("output_dir", None)
    if output_dir is not None and tester_config.get("output_dir", None) is None:
        tester_config["output_dir"] = os.path.join(output_dir, "test_result")

    # add dataloader_config.data_collator
    data_collator = elaina_config.get("data_collator", None)
    if data_collator and "dataloader_config" in tester_config:
        if tester_config["dataloader_config"].get("data_collator", None) is None:
            tester_config["dataloader_config"]["data_collator"] = data_collator


def main():
    args = get_args()
    elaina_config = load_config(args.config)
    initialize_elaina_by_config(elaina_config)
    tester_config = get_tester_config(elaina_config)
    test_dataset = build_dataset(elaina_config.get("test_dataset", None), split="test")
    tester = QwenVLQLoraTester(test_dataset=test_dataset, **tester_config)
    tester.test()


if __name__ == "__main__":
    main()
