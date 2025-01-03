#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

MODEL="/root/autodl-tmp/pretrain-model/Qwen-VL-Chat-Int4" # Qwen/Qwen-VL-Chat-Int4 Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.

# DS_CONFIG_PATH="${PROJECT_DIR}/ds_config/ds_config_zero2.json"
USE_LORA=True
Q_LORA=True
ELAINA_CONFIG_PATH="config/docvqa/qwenvl_docvqa_plain_ocr_lora_ddp_autodl.json"
DS_CONFIG_PATH="ds_config/ds_config_zero2.json"
export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR=localhost
# Remember to use --fp16 instead of --bf16 due to autogptq
# --fix_vit只有在全量微调时才会生效

export PYTHONPATH=$PYTHONPATH:$(pwd)

NUM_NODES=1
if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
    # 如果 CUDA_VISIBLE_DEVICES 有值
    visible_devices=(${CUDA_VISIBLE_DEVICES//,/ })
    NUM_TRAINERS=${#visible_devices[@]}
else
    # 如果 CUDA_VISIBLE_DEVICES 没有值，尝试通过 nvidia-smi 获取所有 GPU 数量
    NUM_TRAINERS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
fi
echo "NUM_TRAINERS: $NUM_TRAINERS"
torchrun  \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes=$NUM_NODES \
    --nproc-per-node=$NUM_TRAINERS \
    qwenvl_qlora_train.py \
    --model_name_or_path $MODEL \
    --elaina_config_path $ELAINA_CONFIG_PATH \
    --fp16 True \
    --fix_vit True \
    --output_dir /root/autodl-tmp/outputs/MPDocVQA/qwenvl_vqa_ocr_output \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --dataloader_num_workers 4 \
    --dataloader_prefetch_factor 2 \
    --save_strategy "steps" \
    --seed 42 \
    --save_steps 1000 \
    --save_total_limit 5 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --gradient_checkpointing \
    --use_lora ${USE_LORA} \
    --q_lora ${Q_LORA} \
    --deepspeed ${DS_CONFIG_PATH}