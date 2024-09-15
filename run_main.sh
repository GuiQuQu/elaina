#!/bin/bash
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

# export MASTER_ADDR=localhost
# export MASTER_PORT=29500

torchrun  \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes=$NUM_NODES \
    --nproc-per-node=$NUM_TRAINERS \
    docvqa_train.py \
    --config_file config/docvqa/internvl2_classify.json