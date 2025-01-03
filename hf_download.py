import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def download_model(repo_id, local_dir, **kwargs):
    from huggingface_hub import snapshot_download # 下载模型

    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        **kwargs,
    )

def download_dataset(repo_id, local_dir, **kwargs):
    from huggingface_hub import snapshot_download # 下载数据集

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        **kwargs,
    )



if __name__ == "__main__":
    download_model(
        repo_id="timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k",
        local_dir="/root/autodl-tmp/pretrain-model/eva02_large_patch14_clip_224.merged2b_s4b_b131k",
        ignore_patterns=["*.safetensors"],
    )
