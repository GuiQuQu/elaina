
from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def train(self):
        pass