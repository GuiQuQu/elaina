from typing import Dict, Any
from torch.utils.data import Dataset

from logger import logger

def prepare_data_and_preprocessor(cls):

    original_init = cls.__init__
    def new_init(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        self.prepare_data_and_preprocessor()
    
    original_prepare_data = cls.prepare_data
    def prepare_data(self):
        data = original_prepare_data(self)
        self.data = data

    cls.__init__ = new_init
    cls.prepare_data = prepare_data
    return cls

def get_cls_or_func(path_str:str):
    parts = path_str.split('.')
    module_path = '.'.join(parts[:-1])
    cls_name = parts[-1]
    logger.debug(f'Loading submodule {cls_name} from module {module_path}')
    module = __import__(module_path, fromlist=[cls_name])
    cls = getattr(module, cls_name)
    return cls

def build_preprocessor(preprocess_config:Dict[str,Any]):
    _type = preprocess_config.pop('type',None)
    if _type == None:
        raise ValueError("preprocess type is not provided")
    cls = get_cls_or_func(_type)
    preprocessor = cls(**preprocess_config)
    return preprocessor


class BaseDataset(Dataset):
    def __init__(self,preprocess_config) -> None:
        super().__init__()
        self.preprocess_config = preprocess_config

    def prepare_data_and_preprocessor(self):
        self.prepare_data()
        self.preprocessor = build_preprocessor(self.preprocess_config)

    def __getitem__(self, index):
        item = self.data[index]
        return self.preprocessor.preprocess(item)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def prepare_data(self):
        self.data = []
        raise NotImplementedError