import json
from typing import Dict, Any
import inspect

import torch

from logger import logger


def get_cls_or_func(path_str: str):
    """
    import a class or function from a string path
    """
    parts = path_str.split(".")
    module_path = ".".join(parts[:-1])
    cls_name = parts[-1]
    logger.debug(f"Loading submodule {cls_name} from module {module_path}")
    module = __import__(module_path, fromlist=[cls_name])
    cls = getattr(module, cls_name)
    return cls

def load_default_config():
    with open('config/default_config.json', 'r') as f:
        config = json.load(f)
    return config

def load_config(config_file):
    """
    Load a json config file
    """
    with open(config_file, "r") as f:
        config = json.load(f)
    defalut_config = load_default_config()
    for k,v in defalut_config.items():
        if k not in config:
            config[k] = v
    return config


def load_state_dict_from_ckpt(ckpt_path, map_location="cpu"):
    """
    Load a pytorch model state dict from a checkpoint file
    """
    if ckpt_path.endswith(".safetensors"):
        import safetensors

        state_dict = safetensors.torch.load_file(
            filename=ckpt_path, device=map_location
        )
        return state_dict
    
    # defaut to torch.load
    state_dict = torch.load(ckpt_path, map_location=map_location)
    return state_dict


def check_model_state_dict_load(model: torch.nn.Module, state_dict):
    """
    Check if all the parameters in the state dict are loaded into the model
    """
    model_keys = set(model.state_dict().keys())
    state_dict_keys = set(state_dict.keys())
    missing_keys = model_keys - state_dict_keys
    unexpected_keys = state_dict_keys - model_keys

    if len(missing_keys) > 0:
        logger.warning(f"[Model load] Missing keys: {missing_keys}")
    if len(unexpected_keys) > 0:
        logger.warning(f"[Model load] Unexpected keys: {unexpected_keys}")
        

def delete_not_used_key_from_batch(model, batch: Dict[str,Any]):
    # 获取函数的参数信息
    signature = inspect.signature(model.forward)

    # 遍历参数并获取参数名
    param_names = []
    for param_name, param in signature.parameters.items():
        param_names.append(param_name)
    param_names = set(param_names)

    # 删除不在参数列表中的key,value对
    delete_dict = {}
    for key in list(batch.keys()):
        if key not in param_names:
            delete_dict[key] = batch[key]
            del batch[key]
    return batch, delete_dict
