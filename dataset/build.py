import torch

from logger import logger
from utils.utils import get_cls_or_func

def build_dataset(dataset_config : dict, split='none') -> torch.utils.data.Dataset:
    if dataset_config is None:
        logger.warning(f'[{split}] No dataset provided.')
        return None
    _type = dataset_config.pop('type', None)
    if _type is None:
        raise ValueError(f'Dataset type not provided in {dataset_config}.')
    return get_cls_or_func(_type)(**dataset_config)