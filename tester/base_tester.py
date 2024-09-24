"""
    Tester需要的内容
    1. test dataset
    2. model
    3. 需要加载的checkpoint列表 ok
    4. data_loader的设置
    5. data_collator的设置
    6. 运行 model的forward函数进行模型的前向(要求可以可以通过继承，来制定如何跑模型)
    7. 模型返回结果的保存 (model预测结果, 真实标签, 然后还有其他可以选择的模型的输入的来源也需要进行保存)
    7. 运行之后的测试结果保存的设置(output_dir)
    8. metrics的设置

    "tester": {
        "type" : "HFTester",
        "checkpoints": [model.pth1, model.path2, ...]
        "hf_checkpoint_dir": ./outputs/MPDocVQA/classify_output
        "dataloader_config" : {
            "batch_size": 32,
            "num_workers": 4,
            "shuffle": False,
            "data_collator": "default_collator"
        },
    }
"""

from abc import ABC, abstractmethod


class BaseTester(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def test():
        pass

    @abstractmethod
    def test_one_checkpoint():
        pass
