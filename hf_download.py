import os

os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

def download_model(repo_id, local_dir, **kwargs):
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        **kwargs,
    )

if __name__ == "__main__":
    download_model(repo_id="Qwen/Qwen2-VL-2B-Instruct", local_dir="/root/autodl-tmp/pretrain-model/Qwen2-VL-2B-Instruct")
