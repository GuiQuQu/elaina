import json
from typing import Dict, Any
import inspect

import torch

from utils.register import Register
from utils.dist_variable import rank0_context
from logger import logger


def get_cls_or_func(path_str_or_name: str):
    """
    Firstly, try find the name whether or not is register,
    if yes, return the register cls or func
    
    Secondly, assume the path_str_or_name is a import moudule path, 
    then import the module and return the cls or func
    
    """
    if path_str_or_name in Register:
        return Register[path_str_or_name]
    else:
        return import_and_get_cls_or_func(path_str_or_name)

def import_and_get_cls_or_func(path_str: str):
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
    Load a json config file, and fill in the missing keys with default values
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

def check_environment():
    """
    Check the environment of the current running code
    """
    import torch
    import platform
    import numpy as np
    import torch
    import torch.cuda
    import torch.backends.cudnn as cudnn

    with rank0_context():
        logger.info(f"Python version: {platform.python_version()}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"PyTorch CUDA version: {torch.version.cuda}")
        logger.info(f"PyTorch cuDNN version: {cudnn.version()}")
        logger.info(f"NumPy version: {np.__version__}")

        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("GPU: Not available")
    
    # check local cuda version
    import torch.utils
    import torch.utils.cpp_extension as ex
    import os
    import re
    import subprocess
    CUDA_HOME = ex.CUDA_HOME

    with rank0_context():
        logger.info(f"CUDA_HOME:{ex.CUDA_HOME}")
    
    if os.path.exists(CUDA_HOME):
        nvcc = os.path.join(CUDA_HOME, 'bin', 'nvcc')
        SUBPROCESS_DECODE_ARGS = ()
        cuda_version_str = subprocess.check_output([nvcc, '--version']).strip().decode(*SUBPROCESS_DECODE_ARGS)
        local_cuda_version = re.search(r'release (\d+[.]\d+)', cuda_version_str)
        with rank0_context():
            logger.info(f"Local CUDA version: {local_cuda_version.group(1)}")
