def check_environment():
    """
    Check the environment of the current running code
    """
    import torch
    import platform
    import numpy as np
    import torch
    import torch.cuda
    import torch.backends.cudnn as cudnn

    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    print(f"PyTorch cuDNN version: {cudnn.version()}")
    print(f"NumPy version: {np.__version__}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU: Not available")

    # check local cuda version
    import torch.utils
    import torch.utils.cpp_extension as ex
    import os
    import re
    import subprocess

    CUDA_HOME = ex.CUDA_HOME

    print(f"CUDA_HOME:{ex.CUDA_HOME}")

    if CUDA_HOME is not None and os.path.exists(CUDA_HOME):
        nvcc = os.path.join(CUDA_HOME, "bin", "nvcc")
        SUBPROCESS_DECODE_ARGS = ()
        cuda_version_str = (
            subprocess.check_output([nvcc, "--version"])
            .strip()
            .decode(*SUBPROCESS_DECODE_ARGS)
        )
        local_cuda_version = re.search(r"release (\d+[.]\d+)", cuda_version_str)

        print(f"Local CUDA version: {local_cuda_version.group(1)}")
    else:
        print(f"Local CUDA version: Not available")

if __name__ == "__main__":
    check_environment()
