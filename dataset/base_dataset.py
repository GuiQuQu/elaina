from typing import Dict, Any, List
from torch.utils.data import Dataset

from utils.utils import get_cls_or_func
from logger import logger


def prepare_data_and_preprocessor(cls):

    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.prepare_data_and_preprocessor()

    original_prepare_data = cls.prepare_data

    def prepare_data(self):
        data = original_prepare_data(self)
        # logger.info(f"Data length: {len(data)}")
        if callable(self.include_func):
            data = list(filter(self.include_func, data))
            # logger.info(f"Data length after filtering: {len(data)}")
        self.data = data

    cls.__init__ = new_init
    cls.prepare_data = prepare_data
    return cls


def build_preprocessor(preprocess_config: Dict[str, Any]):
    _type = preprocess_config.pop("type", None)
    if _type == None:
        raise ValueError("preprocess type is not provided")
    cls = get_cls_or_func(_type)
    preprocessor = cls(**preprocess_config)
    return preprocessor

def get_cls_or_func_no_error(cls_or_func):
    try:
        return get_cls_or_func(cls_or_func)
    except:
        return None


class BaseDataset(Dataset):
    def __init__(self, preprocess_config, include_func=None, exclude_func=None) -> None:
        super().__init__()
        self.preprocess_config = preprocess_config
        exclude_func = get_cls_or_func_no_error(exclude_func)
        include_func = get_cls_or_func_no_error(include_func)
        if exclude_func == None and include_func == None:
            self.include_func = None
        else:
            if include_func == None:
                include_func = lambda item: True
            if exclude_func == None:
                exclude_func = lambda item: False
            self.include_func = lambda item: include_func(item) and not exclude_func(item)

    def prepare_data_and_preprocessor(self):
        self.prepare_data()
        self.preprocessor = build_preprocessor(self.preprocess_config)

    def __getitem__(self, index):
        item = self.data[index]
        return self.preprocessor.preprocess(item)

    def __len__(self) -> int:
        return len(self.data)

    def prepare_data(self) -> List[Dict[str, Any]]:
        raise NotImplementedError
