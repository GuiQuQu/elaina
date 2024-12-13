"""
    提供模型预处理和模型运行并返回结果的函数
"""

import gc
import torch

from trainer.trainer_utils import load_model
from utils.utils import load_state_dict_from_ckpt
from logger import logger
from dataset.base_dataset import build_preprocessor as build_preprocessor_base


def build_preprocessor(preprocess_config: dict):
    if preprocess_config is None:
        return None
    return build_preprocessor_base(preprocess_config)


def load_model_with_checkpoint(
    model_config,
    checkpoint_path: str = None,
    state_dict_map_location: str = "cpu",
):

    if model_config is None:
        return None

    gc.collect()
    torch.cuda.empty_cache()
    model = load_model(model_config)
    if checkpoint_path is not None:
        state_dict = load_state_dict_from_ckpt(checkpoint_path, state_dict_map_location)
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
    if torch.cuda.is_available():
        model = model.to("cuda:0")
    return model


import torch

from tester.custom_tester import delete_not_used_key_from_batch_in_inference


def custom_model_inference(model, preprocessor, input):
    assert (
        getattr(model, "inference_forward", None) is not None
    ), "model should have inference_forward method"

    model_inputs = preprocessor(input)
    model_inputs = delete_not_used_key_from_batch_in_inference(model, model_inputs)
    model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}
    with torch.no_grad():
        _, outputs = model.inference_forward(**model_inputs)
    del model_inputs
    gc.collect()
    torch.cuda.empty_cache()
    return outputs[0]
