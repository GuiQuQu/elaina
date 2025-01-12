#!/bin/bash
# 本地运行
# ssh -CNg -L 7860:127.0.0.1:7860 autodl-2
# ssh -CNg -L 12352:127.0.0.1:7860 autodl-2
# 然后本地访问 127.0.0.1:12352
export PYTHONPATH=$PYTHONPATH:$(pwd)

python3 deploy/run.py