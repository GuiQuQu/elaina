# Introduction

This is a project for [MP-DocVQA](https://rrc.cvc.uab.es/?ch=17&com=evaluation&task=4) tasks. The project is based on the following models
- [InternVL2](https://internvl.github.io/blog/2024-07-02-InternVL-2.0/)
- [Qwen2VL](https://qwenlm.github.io/zh/blog/qwen2-vl/)
- [QwenVL](https://github.com/QwenLM/Qwen-VL)
- [EVA-CLIP](https://github.com/baaivision/EVA/blob/master/EVA-CLIP/README.md)

When only using a **2B size model** and **an RTX 4090 GPU with 24GB memory**, the model achieved **the SOTA ANLS and the SOTA page acc** on the mp-docvqa leaderboard in **Jan 2025**.

Leaderboard is here: [MP-DocVQA](https://rrc.cvc.uab.es/?ch=17&com=evaluation&task=4)

I use a two stage strategy to solve the MP-DocVQA task, based on the dataset's assumption that only one document image can answer the question about document image.

- Step 1. retrieve the most relevant document image from the dataset, using the question and image construct the prompt for MLLM
- Step 2. Use MLLM to do the SP-DocVQA Task

Because of the high page accuracy in Step 1 and the good performance of MLLM in SP-DocVQA, It is foreseeable to achieve the good performance on MP-DocVQA.

Retrieval Model

# Installation

I use conda to manage the environment, you can create a new conda environment by the following command:

```bash
conda create -n elaina python=3.10
```

this is the environment setting for the project:
- Python Version: 3.10
- PyTorch Version: 2.1.2
- CUDA Version: 11.8

Please check file `requirements.txt` for more required packages.

if you use pip to install torch 2.1.2, the default cuda version is 12.1, you need install torch independently by conda.

```bash
# CUDA 11.8
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

```bash
pip install -r requirements.txt
```

> Notice:
>
> If you didn't want to use QwenVL, you can comment the line in `requirements.txt` for qwenvl

# Download Model and Dataset

## Download model
you can use the script `hf_download.py` in repo to download your wanted model in hugingface model hub. please check the script, and modify the `download_model` function to download the model you want.

Args for `download_model` function:
- `repo_id` is the model repo id in hugingface model hub
- `local_dir` is the local model path to save the model
- any other args for `snapshot_download` function, please check the [function document](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/file_download#huggingface_hub.snapshot_download) for more details

enverionment variable in `hf_download.py`:
- the script set `HF_ENDPOINT` env var for China user to use the mirror site of huggingface model hub, you can comment the line to use the official site.

```python
# InternVL2-2B setting:
download_model(
    repo_id="OpenGVLab/InternVL2-2B",
    local_dir="path/to/save/model"
)
# Qwen2VL-2B setting:
download_model(
    repo_id="Qwen/Qwen2-VL-2B-Instruct",
    local_dir="path/to/save/model"
)
# eva-clip setting:
download_model(
    repo_id="timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k",
    local_dir="path/to/save/model",
    ignore_patterns=["*.safetensors"],
)
```

## Download dataset
you can download the MP-DocVQA dataset from the [official site](https://rrc.cvc.uab.es/?ch=17&com=downloads), unzip data and put the dataset in any wanted folder, and organize the dataset as the following structure:

```bash
.
├── images, the image folder
├── ocr, the ocr folder, if you download the ocr data
├── test.json
├── train.json
└── val.json
```

# Training
you can use the script `run_main.sh` to train or eval the model, just provide the config file.

Please cehck **config List** to find your wanted training config

```bash
bash run_main.sh --config config_file --do_train
```

# Evaluation

you can use the script `run_main.sh` to train or eval the model, just provide the config file.

Please cehck **config List** to find your wanted evaluation config

```bash
bash run_main.sh --config config_file --do_test
```

> Notice, all exp which is based on the qwenvl model, plase use the special script to run the exp, the script is in `scripts/qwenvl` directory.

# Config List
This is a config path for model training and evaluation, you can modify the config file to train or eval the model.

Notice , in `config` directory, there is a `default_config.json`, the default config is for all experiments, you can cover the default config in your own config file.

some config file is for training, and you use the config and give the `--do_test`,the exp is to eval the model in eval-dataset.

some config file is for testing, and you use the config and give the `--do_test`,the exp is to test the model in test-dataset.

| No.  | Config File       | Config Description           |
|------|------------------------------------------------------------------------------|---------------------------------------------|
| 1.   | `config/docvqa/spdocvqa_vqa/qwenvl/qwenvl_docvqa_plain_ocr_lora_ddp_autodl.json`  | (train-val) qwen-vl run sp-docvqa w/ ocr, the ocr is plained text       |
| 2.   | `config/docvqa/spdocvqa_vqa/internvl2/internvl2_docvqa_lora_ddp_autodl.json`    | (train-val) internvl2 run sp-docvqa w/o ocr   |
| 3.   | `config/docvqa/spdocvqa_vqa/internvl2/internvl2_docvqa_lora_plain_ocr_ddp_autodlv2.json` | (train-val) internvl2 run sp-docvqa w/ ocr , the ocr is plained text     
| 4.   | `config/docvqa/spdocvqa_vqa/internvl2/internvl2_docvqa_lora_ocr_ddp_autodl.json`   | (train-val) internvl2 run sp-docvqa w/ ocr , the ocr is layouted text          |
| 5.   | `config/docvqa/spdocvqa_vqa/qwen2vl/qwen2vl_docvqa_ddp_lora_autodl.json`   | (train-val) qwen2vl run sp-docvqa w/o ocr  |
| 6.   | `config/docvqa/spdocvqa_vqa/qwen2vl/qwen2vl_docvqa_lora_ocr_autodl.json`   | (train-val) qwen2vl run sp-docvqa w/ ocr, the ocr is layout text   |
| -- | -- | -- |
| 7.   | `config/docvqa/classify/clip/clip_mpdocvqa_classify_autodl.json`  | (train-val) eva-clip run mp-docvqa retrieval                                                     |
| 8.   | `config/docvqa/classify/internvl2/internvl2_mpdocvqa_classify_lora_ddp_autodl.json`  | (train-val) internvl2 + mlp (w/o ocr) run mp-docvqa retrieval and classify   |
| 9.  | `config/docvqa/classify/internvl2/internvl2_mpdocvqa_classify_ab_ddp_autodl.json` | (train-val) internvl2 + ab classify (w/o ocr) run mp-docvqa retrieval and classify  |
| 10.  | `config/docvqa/classify/qwen2vl/qwen2vl_mpdocvqa_classify_ddp_autodl.json` | (train-val) qwen2vl + mlp (w/o ocr) run mp-docvqa retrieval and classify                         |
| 11.  | `config/docvqa/classify/qwen2vl/qwen2vl_mpdocvqa_classify_ab_ddp_autodl.json` | (train-val) qwen2vl + ab classify (w/o ocr) run mp-docvqa retrieval and classify  |
|--| --                  | --     |
|12.| `config/docvqa/mpdocvqa_vqa/train/qwenvl/qwenvl_mpdocvqa_ocr_autodl.json`   | (train) (no run script) qwenvl is used for mp-docvqa vqa task, w/ ocr, ocr is layouted ocr     |
|13.| `config/docvqa/mpdocvqa_vqa/train/internvl2/internvl2_mpdocvqa_vqa_lora_ddp_autodl.json`| (train-eval) internvl2 is used for mp-docvqa vqa task, w/o ocr, base internvl2 classify result, internvl2 is used in mp-docvqa vqa stage, w/o ocr  |
|14.| `config/docvqa/mpdocvqa_vqa/train/internvl2/internvl2_mpdocvqa_vqa_ocr_lora_ddp_autodl.json` | (train-eval) internvl2 is used for mp-docvqa vqa task w/ ocr, ocr is layouted ocr, base internvl2 classify result, internvl2 is used in mp-docvqa vqa stage, w/ ocr, ocr is layouted ocr |
|15.| `config/docvqa/mpdocvqa_vqa/train/qwen2vl/qwen2vl_mpdocvqa_vqa_ddp_autodl.json`| (train-eval) qwen2vl is used for mp-docvqa vqa task, w/o ocr, base internvl2 classify result, qwen2vl is used in mp-docvqa vqa stage, w/o ocr  |
|16.| `config/docvqa/mpdocvqa_vqa/train/qwen2vl/qwen2vl_mpdocvqa_vqa_ocr_ddp_autodl.json`| (train-eval) qwen2vl is used for mp-docvqa vqa task w/ ocr, ocr is layouted ocr,base internvl2 classify result, qwen2vl is used in mp-docvqa vqa stage, w/ ocr, ocr is layouted ocr |
|--|--|--|
|17.| `config/docvqa/mpdocvqa_vqa/base_clip/internvl2_mpdocvqa_vqa_autodl.json` | (eval) base clip classify result, internvl2 is used in mp-docvqa vqa stage, w/o ocr |
|18.| `config/docvqa/mpdocvqa_vqa/base_clip/internvl2_mpdocvqa_vqa_ocr_autodl.json`| (eval) base clip classify result, internvl2 is used in mp-docvqa vqa stage, w/ ocr, ocr is layouted text |
|19.| `config/docvqa/mpdocvqa_vqa/base_clip/qwen2vl_mpdocvqa_vqa_autodl.json`| (eval) base clip classify result, qwen2vl is used in mp-docvqa vqa stage, w/o ocr |
|20.| `config/docvqa/mpdocvqa_vqa/base_clip/qwen2vl_mpdocvqa_vqa_ocr_autodl.json` | (eval) base clip classify result, qwen2vl is used in mp-docvqa vqa stage, w/ ocr, ocr is layouted text |
|--|--|--|
|21.|`config/docvqa/mpdocvqa_vqa/base_qwen2vl/internvl2_mpdocvqa_vqa_ddp_autodl.json`| (eval) base qwen2vl classify result, internvl2 is used in mp-docvqa vqa stage, w/o ocr |
|22.|`config/docvqa/mpdocvqa_vqa/base_qwen2vl/internvl2_mpdocvqa_vqa_ocr_ddp_autodl.json`| (eval) base qwen2vl classify result, internvl2 is used in mp-docvqa vqa stage, w/ ocr, ocr is layouted text|
|23.|`config/docvqa/mpdocvqa_vqa/base_qwen2vl/qwen2vl_mpdocvqa_vqa_autodl.json`| (eval) base qwen2vl classify result, qwen2vl is used in mp-docvqa vqa stage, w/o ocr|
|24.|`config/docvqa/mpdocvqa_vqa/base_qwen2vl/qwen2vl_mpdocvqa_vqa_ocr_autodl.json`| (eval) base qwen2vl classify result, qwen2vl is used in mp-docvqa vqa stage, w/ ocr, ocr is layouted text|
|--|--|--|
|25.|`config/docvqa/testdata_config/clip_mpdocvqa_classify_test_autodl.json`| (test) eva-clip generate mp-docvqa retrieval result in test-dataset |
|26.| `config/docvqa/testdata_config/internvl2_mpdocvqa_classify_lora_ddp_test_autodl.json` | (test) internvl2 + mlp (w/o ocr) generate mp-docvqa retrieval result in test-dataset |
|27.|`config/docvqa/testdata_config/internvl2_mpdocvqa_classify_ab_ddp_test_autodl.json`| (test) internvl2 + a/b classify (w/o ocr) generate mp-docvqa retrieval result in test-dataset|
|28.|`config/docvqa/testdata_config/qwen2vl_mpdocvqa_classify_ddp_test_wty.json`| (test) qwen2vl + mlp (w/o ocr) generate mp-docvqa retrieval result in test-dataset|
|--|--|--|
|29.|`config/docvqa/testdata_config/base_clip/internvl2_mpdocvqa_vqa_test_autodl.json`|(test) base clip classify result, internvl2 is used in mp-docvqa vqa stage, w/o ocr|
|30.|`config/docvqa/testdata_config/base_clip/internvl2_mpdocvqa_vqa_ocr_test_autodl.json`| (test) base clip classify result, internvl2 is used in mp-docvqa vqa stage, w/ ocr, ocr is layouted text|
|31.|`config/docvqa/testdata_config/base_clip/qwen2vl_mpdocvqa_vqa_test_autodl.json`|(test) base clip classify result, qwen2vl is used in mp-docvqa vqa stage, w/o ocr|
|32.|`config/docvqa/testdata_config/base_clip/qwen2vl_mpdocvqa_vqa_ocr_test_autodl.json`|(test) base clip classify result, qwen2vl is used in mp-docvqa vqa stage, w/ ocr, ocr is layouted text|
|--|--|--|
|33.|`config/docvqa/testdata_config/base_internvl2/internvl2_mpdocvqa_vqa_test_autodl.json`|(test) base internvl2 classify result, internvl2 is used in mp-docvqa vqa stage, w/o ocr|
|34.|`config/docvqa/testdata_config/base_internvl2/internvl2_mpdocvqa_vqa_ocr_ddp_test_autodl.json`|(test) base internvl2 classify result, internvl2 is used in mp-docvqa vqa stage, w/ ocr, ocr is layouted text|
|35.|`config/docvqa/testdata_config/base_internvl2/qwen2vl_mpdocvqa_vqa_test_autodl.json`|(test) base internvl2 classify result, qwen2vl is used in mp-docvqa vqa stage, w/o ocr|
|36.|`config/docvqa/testdata_config/base_internvl2/qwen2vl_mpdocvqa_vqa_ocr_test_autodl.json`|(test) base internvl2 classify result, qwen2vl is used in mp-docvqa vqa stage, w/ ocr, ocr is layouted text|
|--|--|--|
|37.|`config/docvqa/testdata_config/base_qwen2vl/internvl2_mpdocvqa_vqa_test_autodl.json`|(test) base qwen2vl classify result, internvl2 is used in mp-docvqa vqa stage, w/o ocr|
|38.|`config/docvqa/testdata_config/base_qwen2vl/internvl2_mpdocvqa_vqa_ocr_ddp_test_autodl.json`|(test) base qwen2vl classify result, internvl2 is used in mp-docvqa vqa stage, w/ ocr, ocr is layouted text|
|39.|`config/docvqa/testdata_config/base_qwen2vl/qwen2vl_mpdocvqa_vqa_test_autodl.json`|(test) base qwen2vl classify result, qwen2vl is used in mp-docvqa vqa stage, w/o ocr|
|40.|`config/docvqa/testdata_config/base_qwen2vl/qwen2vl_mpdocvqa_vqa_ocr_test_autodl.json`|(test) base qwen2vl classify result, qwen2vl is used in mp-docvqa vqa stage, w/ ocr, ocr is layouted text|

