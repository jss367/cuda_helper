import os
import subprocess

import tensorflow as tf


def get_version_via_command(cmd):
    try:
        return subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
    except Exception as e:
        return str(e)


# Get Python version
python_version = os.sys.version

# Get TensorFlow version
tensorflow_version = tf.__version__

# Get CUDA version
cuda_version = get_version_via_command("nvcc --version | grep release | awk '{print $6}' | cut -c2-")

# Get cuDNN version
cudnn_version = get_version_via_command("cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2")
if "No such file or directory" in cudnn_version:
    cudnn_version = "cuDNN not found."

# Get NVIDIA driver version
nvidia_driver_version = get_version_via_command("nvidia-smi | grep -oP 'Driver Version: \K\d+\.\d+'")

# Print all versions
print(f"Python version: {python_version}")
print(f"TensorFlow version: {tensorflow_version}")
print(f"CUDA version: {cuda_version}")
print(f"{cudnn_version}")  # This will print major, minor, and patch version of cuDNN
print(f"NVIDIA Driver version: {nvidia_driver_version}")
