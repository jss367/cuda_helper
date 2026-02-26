import os
import platform
import subprocess
import sys


def get_version_via_command(cmd):
    try:
        return subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
    except Exception as e:
        return str(e)


def get_python_version():
    return sys.version


def get_tensorflow_version():
    try:
        import tensorflow as tf

        version = tf.__version__
    except ImportError:
        version = "Not Found"
    return version


def get_pytorch_version():
    try:
        import torch

        version = torch.__version__
    except ImportError:
        version = "Not Found"
    return version


def get_cuda_version_unix():
    try:
        version = get_version_via_command("nvcc --version | grep release | awk '{print $6}' | cut -c2-")
        success = bool(version and "not found" not in version.lower() and "error" not in version.lower())
    except Exception:
        version = "Not Found"
        success = False
    return version, success


def get_cuda_version_windows():
    """
    Would be better to propagate this error message
    """
    try:
        release_str = get_version_via_command("powershell \"nvcc --version | Select-String 'release'")
        release_splits = release_str.split(" ")
        release_index = release_splits.index("release")
        version = release_splits[release_index + 1]
        success = True
    except Exception:
        version = "Not Found"
        success = False
    return version, success


def get_cudnn_version_unix():
    cudnn_path = get_version_via_command("find /usr/ -name 'cudnn.h' 2>/dev/null")
    if not cudnn_path or "not found" in cudnn_path.lower() or "error" in cudnn_path.lower() or "permission denied" in cudnn_path.lower():
        return None
    # Take only the first result if find returns multiple paths
    cudnn_path = cudnn_path.split("\n")[0].strip()
    if not cudnn_path:
        return None
    cmd = f"grep CUDNN_MAJOR -A 2 {cudnn_path}"
    cudnn_version_info = get_version_via_command(cmd).split("\n")
    if len(cudnn_version_info) >= 3:
        return (
            "cuDNN version:"
            f" {cudnn_version_info[0].split()[-1]}.{cudnn_version_info[1].split()[-1]}.{cudnn_version_info[2].split()[-1]}"
        )
    else:
        return None


def get_nvidia_driver_version():
    output = get_version_via_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits")
    if output:
        # Multi-GPU systems return one line per GPU; driver version is the same, so take the first
        return output.split("\n")[0].strip()
    return output


def get_tf_gpu_availability():
    try:
        import tensorflow as tf

        num_gpus = len(tf.config.experimental.list_physical_devices("GPU"))
    except ImportError:
        num_gpus = 0
    return num_gpus


def get_environment_variables():
    return {"PATH": os.getenv("PATH"), "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH")}


def get_conda_installed_libraries():
    return get_version_via_command("conda list | grep cuda")


def get_operating_system():
    return platform.system()


def print_colored(text, color="green"):
    """
    Print text in the specified color in the terminal.

    Args:
        text (str): The text to print.
        color (str): The color to print the text in. Options are "green" or "red".
    """
    color_code = "\033[32m" if color == "green" else "\033[31m"
    reset_code = "\033[0m"
    print(f"{color_code}{text}{reset_code}")


def main():
    print_colored("\n--- Debug Information ---\n", "green")
    operating_system = get_operating_system()
    print_colored(f"Operating system: {operating_system}\n", "green")
    print_colored(f"Python version: {get_python_version()}\n", "green")
    print_colored(f"TensorFlow version: {get_tensorflow_version()}\n", "green")
    tf_gpu_availability = get_tf_gpu_availability()
    if tf_gpu_availability > 0:
        print_colored(f"TensorFlow GPU available: {tf_gpu_availability} GPUs found.\n", "green")
    else:
        print_colored("TensorFlow GPU available: No GPUs found.\n", "red")

    if operating_system == "Windows":
        cuda_version, cuda_success = get_cuda_version_windows()
    else:
        cuda_version, cuda_success = get_cuda_version_unix()
    if cuda_success:
        print_colored(f"CUDA version: {cuda_version}\n", "green")
    else:
        print_colored("CUDA version: Not found\n", "red")

    if operating_system != "Windows":
        cudnn_version = get_cudnn_version_unix()
        if cudnn_version:
            print_colored(cudnn_version, "green")
        else:
            print_colored("cuDNN version: Not found\n", "red")
    else:
        print_colored("cuDNN version: Detection not supported on Windows\n", "red")

    nvidia_driver_version = get_nvidia_driver_version()
    if nvidia_driver_version:
        print_colored(f"NVIDIA Driver version: {nvidia_driver_version}\n", "green")
    else:
        print_colored("NVIDIA Driver version: Not found\n", "red")

    env_vars = get_environment_variables()
    print_colored(f"PATH: {env_vars['PATH']}\n", "green")
    if env_vars.get("LD_LIBRARY_PATH"):
        print_colored(f"LD_LIBRARY_PATH: {env_vars['LD_LIBRARY_PATH']}\n", "green")
    else:
        print_colored("LD_LIBRARY_PATH: Not set\n", "red")

    conda_libs = get_conda_installed_libraries()
    if conda_libs:
        print_colored("Conda installed CUDA-related libraries:\n", "green")
        print_colored(conda_libs, "green")
    else:
        print_colored("Conda installed CUDA-related libraries: None\n", "red")


if __name__ == "__main__":
    main()
