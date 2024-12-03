export CUDA_HOME=${HOME}/wty/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=0
bash run_main.sh \
    --config config/docvqa/wty/qwen2vl_mpdocvqa_classify_ddp_wty.json  \
    --do_test
