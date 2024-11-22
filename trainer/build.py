
from utils.utils import get_cls_or_func
from trainer.hf_trainer import HFTrainer, build_hf_trainer


def build_trainer(config):
    trainer_type = config.get('trainer', None)
    if trainer_type is None:
        raise ValueError('No trainer provided.')
    
    cls = get_cls_or_func(trainer_type)
    # if cls == HFTrainer:
    #     return build_hf_trainer(config)
    if issubclass(cls,HFTrainer):
        return build_hf_trainer(config)
    else:
        raise ValueError(f'Unsupported trainer type {trainer_type}.')
    

from abc import ABCMeta, abstractmethod