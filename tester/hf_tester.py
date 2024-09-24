import os

from tester.default_tester import DefaultTester

TRARGET_FILE_NAMES = ["pytorch_model.bin", "model.safetensors"]


class HFTester(DefaultTester):
    def __init__(
        self,
        model_config,
        test_dataset,
        output_dir: str,
        dataloader_config,
        hf_checkpoint_dir: str,
    ):
        super().__init__(
            model_config,
            test_dataset,
            output_dir,
            dataloader_config,
        )
        # add checkpoint_list
        self.checkpoint_list.append(self._get_checkpoint_list(hf_checkpoint_dir))

    def _get_checkpoint_list(self, hf_checkpoint_dir):
        result = []
        for item in os.listdir(hf_checkpoint_dir):
            full_path = os.path.join(hf_checkpoint_dir, item)
            if os.path.isdir(full_path):
                result.append(self._check_hf_checkpoint(full_path))

    def _check_hf_checkpoint(self, checkpoint_path):
        for item in os.listdir(checkpoint_path):
            if item in TRARGET_FILE_NAMES:
                return os.path.join(checkpoint_path, item)
        raise ValueError(
            f"Can not find checkpoint in {checkpoint_path}, allowed files are {TRARGET_FILE_NAMES}"
        )

