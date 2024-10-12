from abc import ABC, abstractmethod
import os
import json
from typing import Tuple


def load_json(json_path):
    with open(json_path, "r") as f:
        result = json.load(f)
    return result


class BaseMetrics(ABC):
    def __init__(self, result_path):
        self.result_path = result_path
        self.result = self.load_result()

    @abstractmethod
    def compute_metrics(self) -> Tuple[float, str]:
        """
            Return the metrics value and the details of the value
        """
        raise NotImplementedError

    def load_result(self):
        if os.path.isfile(self.result_path):
            return load_json(self.result_path)
        elif os.path.isdir(self.result_path):
            json_files = get_json_files(self.result_path)
            result = []
            for json_file in json_files:
                result.extend(load_json(json_file))
            return result


def get_json_files(folder_path):
    json_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files
