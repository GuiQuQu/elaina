import abc
from typing import List


class BasePreprocessor(abc.ABC):
    def __init__(self) -> None:
        self.save_keys = []

    @abc.abstractmethod
    def preprocess(self, item):
        pass
