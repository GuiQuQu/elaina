from typing import Any, List, Tuple
from tester.default_tester import DefaultTester


class GenerateTester(DefaultTester):
    def __init__(
        self,
        model_config,
        test_dataset,
        output_dir,
        dataloader_config,
        metrics,
        max_steps: int = -1,
        checkpoint_list: List[str] = ...,
    ):
        super().__init__(
            model_config,
            test_dataset,
            output_dir,
            dataloader_config,
            metrics,
            max_steps,
            checkpoint_list,
        )

    def run_model(self, model, batch: dict) -> Tuple[float | List[Any]]:
        pass
        # intenvl2 需要特殊的generate传参,这个方法还是无法推广