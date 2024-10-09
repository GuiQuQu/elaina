from copy import deepcopy

from utils.utils import get_cls_or_func


def build_metrics(metrics_config, result_path):
    metrics_config = deepcopy(metrics_config)
    _type = metrics_config.pop("type", None)
    if _type == None:
        raise ValueError("metrics type is not provided")
    cls = get_cls_or_func(_type)

    return cls(result_path=result_path, **metrics_config)
