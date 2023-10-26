import subprocess
import sys

import tensorflow as tf


def get_version_via_command(cmd):
    try:
        return subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
    except Exception as e:
        return str(e)


# Get Python version
python_version = sys.version

# Get TensorFlow version
tensorflow_version = tf.__version__

# # Get CUDA version
# cuda_version = get_version_via_command("nvcc --version | grep release | awk '{print $6}' | cut -c2-")

# Check CUDA version
try:
    cuda_version = subprocess.check_output(["nvcc", "--version"]).decode("utf-8").split("V")[-1].strip()
    print("CUDA version:", cuda_version)
except Exception as e:
    print("Error getting CUDA version:", str(e))


# # Get cuDNN version
# cudnn_version = get_version_via_command("cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2")
# if "No such file or directory" in cudnn_version:
#     cudnn_version = "cuDNN not found."


# Check cuDNN version
try:
    cudnn_path = subprocess.check_output(["find", "/usr/", "-name", "cudnn.h"]).decode("utf-8").strip().split("\n")[0]
    if cudnn_path:
        cudnn_version = subprocess.check_output(["cat", cudnn_path, "|", "grep", "CUDNN_MAJOR", "-A", "2"]).decode(
            "utf-8"
        )
        print(
            "cuDNN version:",
            cudnn_version.split("\n")[0].split()[-1]
            + "."
            + cudnn_version.split("\n")[1].split()[-1]
            + "."
            + cudnn_version.split("\n")[2].split()[-1],
        )
    else:
        print("cuDNN not found!")
except Exception as e:
    print("Error getting cuDNN version:", str(e))


# # Get NVIDIA driver version
# nvidia_driver_version = get_version_via_command("nvidia-smi | grep -oP 'Driver Version: \K\d+\.\d+'")

# Check NVIDIA driver version
try:
    nvidia_driver_version = (
        subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"])
        .decode("utf-8")
        .strip()
    )
    print("NVIDIA Driver version:", nvidia_driver_version)
except Exception as e:
    print("Error getting NVIDIA driver version:", str(e))


# Print all versions
print(f"Python version: {python_version}")
print(f"TensorFlow version: {tensorflow_version}")
print(f"CUDA version: {cuda_version}")
print(f"{cudnn_version}")  # This will print major, minor, and patch version of cuDNN
print(f"NVIDIA Driver version: {nvidia_driver_version}")
