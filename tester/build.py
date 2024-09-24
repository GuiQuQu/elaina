import os

from utils.utils import get_cls_or_func
from trainer.trainer_utils import load_model, load_dataset


def build_tester(config):
    tester_config = config.get("tester", None)
    if tester_config is None:
        raise ValueError("Tester config is not provided in the config file.")
    _type = tester_config.pop("type", None)
    if _type == None:
        raise ValueError("Tester type is not provided in the config file.")
    tester_cls = get_cls_or_func(_type)

    model_config = config.get("model", None)
    test_dataset = load_dataset(config.get("test_dataset", None), split="test")

    # add output_dir
    output_dir = config.get("output_dir", None)
    if output_dir is not None and tester_config.get("output_dir", None) is None:
        tester_config["output_dir"] = os.path.join(output_dir, "test_result")

    # add dataloader_config.data_collator
    if "data_collator" in config and "dataloader_config" in tester_config:
        if tester_config["dataloader_config"].get("data_collator", None) is None:
            tester_config["dataloader_config"]["data_collator"] = get_cls_or_func(
                config["data_collator"]
            )
    
    tester = tester_cls(
        model_config=model_config, testdataset=test_dataset, **tester_config
    )
    return tester
