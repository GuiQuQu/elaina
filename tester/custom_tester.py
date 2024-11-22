import os
import inspect
from typing import Any, List, Tuple, Dict
import json
from tester.default_tester import DefaultTester

from utils.register import Register


def delete_not_used_key_from_batch_in_inference(model, batch: Dict[str, Any]):
    # 获取函数的参数信息
    signature = inspect.signature(model.inference_forward)

    # 遍历参数并获取参数名
    param_names = []
    for param_name, param in signature.parameters.items():
        param_names.append(param_name)
    param_names = set(param_names)

    # 删除不在参数列表中的key,value对
    delete_dict = {}
    for key in list(batch.keys()):
        if key not in param_names:
            delete_dict[key] = batch[key]
            del batch[key]
    return batch, delete_dict


@Register(name="custom_tester")
class CustomTester(DefaultTester):
    def __init__(
        self,
        # 自动传入的内容
        model_config,
        test_dataset,
        # 需要在配置文件中填写的内容
        output_dir,
        dataloader_config,
        metrics=[],
        checkpoint_list: List[str] = [],
    ):
        super().__init__(
            model_config=model_config,
            test_dataset=test_dataset,
            output_dir=output_dir,
            dataloader_config=dataloader_config,
            metrics=metrics,
            max_steps=-1,
            checkpoint_list=checkpoint_list,
        )

    def run_model(self, model, batch: dict) -> Tuple[float | List[Any]]:
        """
        在model内部定义inference_forward函数, 该函数的返回类型
        Tuple[float, List[Any]] 为模型推理的输出
        """
        model_input_batch, delete_batch = delete_not_used_key_from_batch_in_inference(
            model, batch
        )
        outputs = model.inference_forward(**model_input_batch)
        batch.update(delete_batch)
        return outputs
