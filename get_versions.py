import os
import subprocess
import sys

import tensorflow as tf


def get_version_via_command(cmd):
    try:
        return subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
    except Exception as e:
        return str(e)


def get_python_version():
    return sys.version


def get_tensorflow_version():
    return tf.__version__


def get_cuda_version():
    return get_version_via_command("nvcc --version | grep release | awk '{print $6}' | cut -c2-")


def get_cudnn_version():
    cudnn_path = get_version_via_command("find /usr/ -name 'cudnn.h'")
    if "No such file or directory" not in cudnn_path:
        cmd = f"grep CUDNN_MAJOR -A 2 {cudnn_path}"
        cudnn_version_info = get_version_via_command(cmd).split("\n")
        if len(cudnn_version_info) >= 3:
            return (
                "cuDNN version:"
                f" {cudnn_version_info[0].split()[-1]}.{cudnn_version_info[1].split()[-1]}.{cudnn_version_info[2].split()[-1]}"
            )
        else:
            return "Error retrieving cuDNN version details"
    else:
        return "cuDNN not found or permission denied."


def get_nvidia_driver_version():
    return get_version_via_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits")


def get_tf_gpu_availability():
    return len(tf.config.experimental.list_physical_devices("GPU"))


def get_environment_variables():
    return {"PATH": os.getenv("PATH"), "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH")}


def get_conda_installed_libraries():
    return get_version_via_command("conda list | grep cuda")


def main():
    print(f"Python version: {get_python_version()}")
    print(f"TensorFlow version: {get_tensorflow_version()}")
    print(f"TensorFlow GPU available: {get_tf_gpu_availability()} GPUs found.")
    print(f"CUDA version: {get_cuda_version()}")
    print(get_cudnn_version())
    print(f"NVIDIA Driver version: {get_nvidia_driver_version()}")
    env_vars = get_environment_variables()
    print(f"PATH: {env_vars['PATH']}")
    print(f"LD_LIBRARY_PATH: {env_vars['LD_LIBRARY_PATH']}")
    print("Conda installed CUDA-related libraries:")
    print(get_conda_installed_libraries())


if __name__ == "__main__":
    main()
