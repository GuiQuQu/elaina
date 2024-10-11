from collections import defaultdict
import os
import json
from typing import Any, Dict, List


from dataset.base_dataset import BaseDataset, prepare_data_and_preprocessor


def open_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["data"]