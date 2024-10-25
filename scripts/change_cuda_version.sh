# 修改cuda的版本(11.8 -> 12.1 or 12.1 to 11.8)
TARGET_CUDA_VERSION=11.8

TARGET_CUDA_HOME=/usr/local/cuda-${TARGET_CUDA_VERSION}

CUDA_HOME=/usr/local/cuda

unlink ${CUDA_HOME} && ln -s ${TARGET_CUDA_HOME} ${CUDA_HOME}
${CUDA_HOME}/bin/nvcc -V