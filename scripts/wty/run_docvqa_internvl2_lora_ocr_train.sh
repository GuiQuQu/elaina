export CUDA_HOME=${HOME}/wty/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=1
bash run_main.sh \
    --config config/docvqa/wty/internvl2_docvqa_lora_ocr_wty.json  \
    --do_train