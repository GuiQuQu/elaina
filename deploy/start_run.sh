#!/bin/bash
# 本地运行
# ssh -CNg -L 7860:127.0.0.1:7860 autodl-2
# 然后本地访问 127.0.0.1：7860
export PYTHONPATH=$PYTHONPATH:$(pwd)

python3 deploy/run.py