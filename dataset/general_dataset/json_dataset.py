import os
import json

from dataset.base_dataset import BaseDataset, prepare_data_and_preprocessor
from utils.register import Register




@Register(name="json_dataset")
@prepare_data_and_preprocessor
class JsonDataset(BaseDataset):
    def __init__(self, preprocess_config, json_path: str) -> None:
        super().__init__(preprocess_config)
        self.json_path = json_path

    def prepare_data(self):
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)
        return data
