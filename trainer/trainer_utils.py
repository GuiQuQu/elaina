import os
import torch
import transformers

from logger import logger
from utils.utils import get_cls_or_func, load_config

def load_dataset(dataset_config, split='none') -> torch.utils.data.Dataset:
    if dataset_config is None:
        logger.warning(f'[{split}] No dataset provided.')
        return None
    _type = dataset_config.pop('type', None)
    if _type is None:
        raise ValueError(f'Dataset type not provided in {dataset_config}.')
    return get_cls_or_func(_type)(**dataset_config)


def load_model(model_config) -> torch.nn.Module:
    if model_config is None:
        raise ValueError('No model provided.')
    _type = model_config.pop('type', None)
    if _type is None:
        raise ValueError(f'Model type not provided in {model_config}.')
    return get_cls_or_func(_type)(**model_config)



    





