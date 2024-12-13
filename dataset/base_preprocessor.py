import abc
from typing import List
from utils.register import Register

class BasePreprocessor(abc.ABC):
    def __init__(self) -> None:
        self.save_keys = []

    @abc.abstractmethod
    def preprocess(self, item):
        pass


@Register(name='unchanged_preprocessor')
class UnchangedPreprocessor(BasePreprocessor):
    def preprocess(self, item):
        return item
