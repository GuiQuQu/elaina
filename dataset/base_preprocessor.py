import abc
from typing import List


class BasePreprocessor(abc.ABC):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def preprocess(self, item):
        pass
