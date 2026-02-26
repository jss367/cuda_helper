import argparse
import json
import os
import platform
import subprocess
import sys


def get_version_via_command(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode("utf-8").strip()
    except Exception:
        return None


def get_python_version():
    return sys.version


def get_tensorflow_version():
    # Run in subprocess to suppress TensorFlow's verbose stderr warnings
    script = "try:\n    import tensorflow as tf; print(tf.__version__)\nexcept ImportError:\n    print('Not Found')"
    try:
        output = subprocess.check_output(
            [sys.executable, "-c", script], stderr=subprocess.DEVNULL, timeout=30
        ).decode("utf-8").strip()
        return output
    except Exception:
        return "Not Found"


def get_pytorch_version():
    # Run in subprocess to avoid CUDA conflicts with TensorFlow
    script = "try:\n    import torch; print(torch.__version__)\nexcept ImportError:\n    print('Not Found')"
    try:
        output = subprocess.check_output(
            [sys.executable, "-c", script], stderr=subprocess.DEVNULL, timeout=30
        ).decode("utf-8").strip()
        return output
    except Exception:
        return "Not Found"


def get_pytorch_gpu_availability():
    # Run in subprocess to avoid CUDA conflicts with TensorFlow
    script = (
        "import json, torch; "
        "a = torch.cuda.is_available(); "
        "n = torch.cuda.device_count() if a else 0; "
        "d = [torch.cuda.get_device_name(i) for i in range(n)]; "
        "print(json.dumps({'available': a, 'device_count': n, 'devices': d}))"
    )
    try:
        output = subprocess.check_output(
            [sys.executable, "-c", script], stderr=subprocess.DEVNULL, timeout=30
        ).decode("utf-8").strip()
        return json.loads(output)
    except Exception:
        return {"available": False, "device_count": 0, "devices": []}


def get_pytorch_mps_availability():
    # Run in subprocess to avoid import conflicts
    script = (
        "import json, torch; "
        "b = torch.backends.mps.is_built(); "
        "a = torch.backends.mps.is_available() if b else False; "
        "print(json.dumps({'available': a, 'is_built': b}))"
    )
    try:
        output = subprocess.check_output(
            [sys.executable, "-c", script], stderr=subprocess.DEVNULL, timeout=30
        ).decode("utf-8").strip()
        return json.loads(output)
    except Exception:
        return {"available": False, "is_built": False}


def get_jax_info():
    # Run in subprocess to avoid CUDA conflicts with other frameworks
    script = (
        "import json; "
        "try:\n"
        "    import jax; v = jax.__version__\n"
        "    try:\n"
        "        g = len(jax.devices('gpu'))\n"
        "    except RuntimeError:\n"
        "        g = 0\n"
        "    print(json.dumps({'version': v, 'gpu_count': g}))\n"
        "except ImportError:\n"
        "    print(json.dumps({'version': 'Not Found', 'gpu_count': 0}))"
    )
    try:
        output = subprocess.check_output(
            [sys.executable, "-c", script], stderr=subprocess.DEVNULL, timeout=30
        ).decode("utf-8").strip()
        return json.loads(output)
    except Exception:
        return {"version": "Not Found", "gpu_count": 0}


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
    cudnn_version_info = get_version_via_command(cmd)
    if not cudnn_version_info:
        return None
    cudnn_version_info = cudnn_version_info.split("\n")
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


def get_gpu_info():
    output = get_version_via_command(
        "nvidia-smi --query-gpu=name,memory.total,utilization.gpu,temperature.gpu "
        "--format=csv,noheader,nounits"
    )
    if not output or "error" in output.lower() or "not found" in output.lower():
        return []
    gpus = []
    for line in output.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            gpus.append({
                "name": parts[0],
                "vram_mb": parts[1],
                "utilization_pct": parts[2],
                "temperature_c": parts[3],
            })
    return gpus


def get_apple_gpu_info():
    try:
        output = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"], stderr=subprocess.DEVNULL, timeout=15
        ).decode("utf-8")
    except Exception:
        return []
    gpus = []
    current = {}
    for line in output.split("\n"):
        stripped = line.strip()
        if stripped.startswith("Chipset Model:"):
            if current:
                gpus.append(current)
            current = {"name": stripped.split(":", 1)[1].strip()}
        elif stripped.startswith("Total Number of Cores:") and current:
            current["cores"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("VRAM") and current:
            current["memory"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Memory:") and "memory" not in current and current:
            current["memory"] = stripped.split(":", 1)[1].strip()
    if current:
        gpus.append(current)
    return gpus


def get_cuda_version_from_nvidia_smi():
    output = get_version_via_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
    if not output or "error" in output.lower() or "not found" in output.lower():
        return None
    # nvidia-smi prints the max CUDA version in its header; parse it from the full output
    full_output = get_version_via_command("nvidia-smi")
    if not full_output or "error" in full_output.lower():
        return None
    for line in full_output.split("\n"):
        if "CUDA Version" in line:
            # Line looks like: | NVIDIA-SMI 535.129.03   Driver Version: 535.129.03   CUDA Version: 12.2  |
            parts = line.split("CUDA Version:")
            if len(parts) >= 2:
                return parts[1].strip().rstrip("|").strip()
    return None


def get_tf_gpu_availability():
    # Run in subprocess to suppress TensorFlow's verbose stderr warnings
    script = (
        "try:\n"
        "    import tensorflow as tf\n"
        "    print(len(tf.config.experimental.list_physical_devices('GPU')))\n"
        "except ImportError:\n"
        "    print(0)"
    )
    try:
        output = subprocess.check_output(
            [sys.executable, "-c", script], stderr=subprocess.DEVNULL, timeout=60
        ).decode("utf-8").strip()
        return int(output)
    except Exception:
        return 0


def get_environment_variables():
    return {"PATH": os.getenv("PATH"), "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH")}


def get_conda_installed_libraries():
    return get_version_via_command("conda list | grep cuda")


def get_pip_cuda_packages():
    output = get_version_via_command("pip list 2>/dev/null")
    if not output or "error" in output.lower():
        return None
    lines = []
    for line in output.split("\n"):
        lower = line.lower()
        if "cuda" in lower or "nvidia" in lower or "cudnn" in lower or "tensorrt" in lower:
            lines.append(line.strip())
    return "\n".join(lines) if lines else None


def get_operating_system():
    return platform.system()


def _extract_pip_cuda_major(pip_packages_str):
    """Extract the set of CUDA major versions targeted by pip packages."""
    majors = set()
    if not pip_packages_str:
        return majors
    for line in pip_packages_str.split("\n"):
        lower = line.lower()
        for token in lower.replace("-", " ").replace("_", " ").split():
            if token.startswith("cu") and len(token) >= 3 and token[2:].isdigit():
                majors.add(token[2:])
    return majors


def _extract_conda_cuda_major(conda_packages_str):
    """Extract CUDA major version from conda cudatoolkit package."""
    if not conda_packages_str:
        return None
    for line in conda_packages_str.split("\n"):
        parts = line.split()
        if parts and parts[0] == "cudatoolkit" and len(parts) >= 2:
            return parts[1].split(".")[0]
    return None


def _extract_path_cuda_major(env_path):
    """Extract CUDA major version from cuda paths like /usr/local/cuda-11.8/bin."""
    import re
    if not env_path:
        return None
    match = re.search(r"/cuda[/-](\d+)\.\d+", env_path)
    if match:
        return match.group(1)
    return None


def get_compatibility_summary(nvcc_version, nvidia_smi_version, pip_packages_str,
                              conda_packages_str, env_path, env_ld_path):
    warnings = []
    pip_majors = _extract_pip_cuda_major(pip_packages_str)

    # Collect all system-level CUDA major versions we can find
    system_sources = {}

    if nvcc_version:
        nvcc_clean = nvcc_version.rstrip(",").strip()
        nvcc_major = nvcc_clean.split(".")[0]
        if nvcc_major.isdigit():
            system_sources[f"nvcc ({nvcc_clean})"] = nvcc_major

    conda_major = _extract_conda_cuda_major(conda_packages_str)
    if conda_major:
        system_sources[f"conda cudatoolkit"] = conda_major

    path_major = _extract_path_cuda_major(env_path)
    if path_major:
        system_sources["PATH"] = path_major

    ld_major = _extract_path_cuda_major(env_ld_path)
    if ld_major:
        system_sources["LD_LIBRARY_PATH"] = ld_major

    # Compare each system source against pip package CUDA versions
    if pip_majors:
        pip_major_str = "/".join(sorted(pip_majors))
        for source_label, source_major in system_sources.items():
            if source_major not in pip_majors:
                warnings.append(
                    f"{source_label} targets CUDA {source_major} "
                    f"but pip packages target CUDA {pip_major_str}"
                )

    # Check for mixed CUDA versions across system sources themselves
    unique_system_majors = set(system_sources.values())
    if len(unique_system_majors) > 1:
        parts = [f"{label} = CUDA {ver}" for label, ver in system_sources.items()]
        warnings.append(f"Mixed system CUDA versions detected: {', '.join(parts)}")

    # Check nvidia-smi max CUDA vs pip packages (driver compatibility, not a mismatch)
    # nvidia-smi reports max supported version, so pip targeting a lower version is fine
    if nvidia_smi_version and pip_majors:
        smi_major = nvidia_smi_version.split(".")[0]
        for pm in pip_majors:
            if pm.isdigit() and smi_major.isdigit() and int(pm) > int(smi_major):
                warnings.append(
                    f"pip packages target CUDA {pm} but driver only supports up to CUDA {nvidia_smi_version}"
                )

    return warnings


def print_colored(text, color="green"):
    """
    Print text in the specified color in the terminal.

    Args:
        text (str): The text to print.
        color (str): The color to print the text in. Options are "green", "red", or "yellow".
    """
    colors = {"green": "\033[32m", "red": "\033[31m", "yellow": "\033[33m"}
    color_code = colors.get(color, "\033[32m")
    reset_code = "\033[0m"
    print(f"{color_code}{text}{reset_code}")


def print_health_summary(checks):
    print_colored("\n--- Health Summary ---\n", "green")
    for label, status in checks:
        if status == "PASS":
            print_colored(f"  [PASS] {label}", "green")
        elif status == "WARN":
            print_colored(f"  [WARN] {label}", "yellow")
        else:
            print_colored(f"  [FAIL] {label}", "red")
    print()


def main():
    parser = argparse.ArgumentParser(description="CUDA/GPU diagnostic tool")
    parser.add_argument("--json", action="store_true", dest="json_mode", help="Output results as JSON")
    args = parser.parse_args()

    data = {}
    health_checks = []

    # 1. OS and Python version
    operating_system = get_operating_system()
    data["operating_system"] = operating_system
    data["python_version"] = get_python_version()
    if not args.json_mode:
        print_colored("\n--- Debug Information ---\n", "green")
        print_colored(f"Operating system: {operating_system}\n", "green")
        print_colored(f"Python version: {data['python_version']}\n", "green")

    # 2. TensorFlow version + GPU count
    tf_version = get_tensorflow_version()
    tf_gpu_count = get_tf_gpu_availability()
    data["tensorflow_version"] = tf_version
    data["tensorflow_gpu_count"] = tf_gpu_count
    if not args.json_mode:
        print_colored(f"TensorFlow version: {tf_version}\n", "green")
        if tf_gpu_count > 0:
            print_colored(f"TensorFlow GPU available: {tf_gpu_count} GPUs found.\n", "green")
        else:
            print_colored("TensorFlow GPU available: No GPUs found.\n", "red")
    health_checks.append(("TensorFlow GPU", "PASS" if tf_gpu_count > 0 else "WARN"))

    # 3. PyTorch version + GPU availability
    pytorch_version = get_pytorch_version()
    pytorch_gpu = get_pytorch_gpu_availability()
    data["pytorch_version"] = pytorch_version
    data["pytorch_gpu"] = pytorch_gpu
    if not args.json_mode:
        print_colored(f"PyTorch version: {pytorch_version}\n", "green")
        if pytorch_gpu["available"]:
            device_names = ", ".join(pytorch_gpu["devices"]) if pytorch_gpu["devices"] else "unknown"
            print_colored(
                f"PyTorch GPU available: {pytorch_gpu['device_count']} GPUs found ({device_names}).\n",
                "green",
            )
        else:
            print_colored("PyTorch GPU available: No GPUs found.\n", "red")
    health_checks.append(("PyTorch GPU", "PASS" if pytorch_gpu["available"] else "WARN"))

    # 3b. PyTorch MPS availability (macOS Apple Silicon)
    if operating_system == "Darwin":
        pytorch_mps = get_pytorch_mps_availability()
        data["pytorch_mps"] = pytorch_mps
        if not args.json_mode:
            if pytorch_mps["available"]:
                print_colored("PyTorch MPS (Apple Silicon GPU): Available\n", "green")
            elif pytorch_mps["is_built"]:
                print_colored("PyTorch MPS (Apple Silicon GPU): Built but not available\n", "yellow")
            else:
                print_colored("PyTorch MPS (Apple Silicon GPU): Not available\n", "red")
        health_checks.append(("PyTorch MPS", "PASS" if pytorch_mps["available"] else "WARN"))

    # 4. JAX version + GPU availability
    jax_info = get_jax_info()
    data["jax"] = jax_info
    if not args.json_mode:
        print_colored(f"JAX version: {jax_info['version']}\n", "green")
        if jax_info["gpu_count"] > 0:
            print_colored(f"JAX GPU available: {jax_info['gpu_count']} GPUs found.\n", "green")
        elif jax_info["version"] != "Not Found":
            print_colored("JAX GPU available: No GPUs found.\n", "red")

    # 5. CUDA version (nvcc)
    if operating_system == "Windows":
        cuda_version, cuda_success = get_cuda_version_windows()
    else:
        cuda_version, cuda_success = get_cuda_version_unix()
    data["cuda_version_nvcc"] = cuda_version if cuda_success else None
    if not args.json_mode:
        if cuda_success:
            print_colored(f"CUDA version (nvcc): {cuda_version}\n", "green")
        else:
            print_colored("CUDA version (nvcc): Not found\n", "red")
    health_checks.append(("CUDA nvcc", "PASS" if cuda_success else "FAIL"))

    # 6. CUDA version (nvidia-smi)
    cuda_smi_version = get_cuda_version_from_nvidia_smi()
    data["cuda_version_nvidia_smi"] = cuda_smi_version
    if not args.json_mode:
        if cuda_smi_version:
            print_colored(f"CUDA version (nvidia-smi): {cuda_smi_version}\n", "green")
        else:
            print_colored("CUDA version (nvidia-smi): Not found\n", "red")
    health_checks.append(("CUDA nvidia-smi", "PASS" if cuda_smi_version else "FAIL"))

    # 7. cuDNN version
    if operating_system != "Windows":
        cudnn_version = get_cudnn_version_unix()
        data["cudnn_version"] = cudnn_version
        if not args.json_mode:
            if cudnn_version:
                print_colored(f"{cudnn_version}\n", "green")
            else:
                print_colored("cuDNN version: Not found\n", "red")
    else:
        data["cudnn_version"] = None
        if not args.json_mode:
            print_colored("cuDNN version: Detection not supported on Windows\n", "red")

    # 8. NVIDIA driver version
    nvidia_driver_version = get_nvidia_driver_version()
    data["nvidia_driver_version"] = nvidia_driver_version
    if not args.json_mode:
        if data["nvidia_driver_version"]:
            print_colored(f"NVIDIA Driver version: {data['nvidia_driver_version']}\n", "green")
        else:
            print_colored("NVIDIA Driver version: Not found\n", "red")
    health_checks.append(("NVIDIA Driver", "PASS" if data["nvidia_driver_version"] else "FAIL"))

    # 9. GPU hardware info
    gpu_info = get_gpu_info()
    data["gpus"] = gpu_info
    apple_gpus = []
    if not gpu_info and operating_system == "Darwin":
        apple_gpus = get_apple_gpu_info()
        data["apple_gpus"] = apple_gpus
    if not args.json_mode:
        if gpu_info:
            print_colored("GPU Hardware Info:\n", "green")
            for i, gpu in enumerate(gpu_info):
                print_colored(
                    f"  GPU {i}: {gpu['name']} | VRAM: {gpu['vram_mb']} MB | "
                    f"Utilization: {gpu['utilization_pct']}% | Temp: {gpu['temperature_c']}°C",
                    "green",
                )
            print()
        elif apple_gpus:
            print_colored("Apple GPU Hardware Info:\n", "green")
            for i, gpu in enumerate(apple_gpus):
                parts = [gpu["name"]]
                if "cores" in gpu:
                    parts.append(f"Cores: {gpu['cores']}")
                if "memory" in gpu:
                    parts.append(f"Memory: {gpu['memory']}")
                print_colored(f"  GPU {i}: {' | '.join(parts)}", "green")
            print()
        else:
            print_colored("GPU Hardware Info: Not available\n", "red")

    # 10. PATH and LD_LIBRARY_PATH
    env_vars = get_environment_variables()
    data["path"] = env_vars["PATH"]
    data["ld_library_path"] = env_vars.get("LD_LIBRARY_PATH")
    if not args.json_mode:
        print_colored(f"PATH: {env_vars['PATH']}\n", "green")
        if env_vars.get("LD_LIBRARY_PATH"):
            print_colored(f"LD_LIBRARY_PATH: {env_vars['LD_LIBRARY_PATH']}\n", "green")
        else:
            print_colored("LD_LIBRARY_PATH: Not set\n", "red")

    # 11. Conda CUDA packages
    conda_libs = get_conda_installed_libraries()
    has_conda_libs = bool(conda_libs and "error" not in conda_libs.lower() and "not found" not in conda_libs.lower())
    data["conda_cuda_packages"] = conda_libs if has_conda_libs else None
    if not args.json_mode:
        if has_conda_libs:
            print_colored("Conda installed CUDA-related libraries:\n", "green")
            print_colored(conda_libs + "\n", "green")
        else:
            print_colored("Conda installed CUDA-related libraries: None\n", "red")

    # 12. pip CUDA packages
    pip_packages = get_pip_cuda_packages()
    data["pip_cuda_packages"] = pip_packages
    if not args.json_mode:
        if pip_packages:
            print_colored("pip installed CUDA-related packages:\n", "green")
            print_colored(pip_packages + "\n", "green")
        else:
            print_colored("pip installed CUDA-related packages: None\n", "red")

    # 13. Compatibility check
    compat_warnings = get_compatibility_summary(
        nvcc_version=cuda_version if cuda_success else None,
        nvidia_smi_version=cuda_smi_version,
        pip_packages_str=pip_packages,
        conda_packages_str=conda_libs if has_conda_libs else None,
        env_path=env_vars.get("PATH"),
        env_ld_path=env_vars.get("LD_LIBRARY_PATH"),
    )
    data["compatibility_warnings"] = compat_warnings
    if not args.json_mode:
        if compat_warnings:
            print_colored("\n--- Compatibility Warnings ---\n", "yellow")
            for warn in compat_warnings:
                print_colored(f"  [WARN] {warn}", "yellow")
            print()
        else:
            print_colored("Compatibility check: No issues detected.\n", "green")
    if compat_warnings:
        health_checks.append(("Compatibility", "WARN"))
    else:
        health_checks.append(("Compatibility", "PASS"))

    # 14. Output
    if args.json_mode:
        # Add health checks to JSON output
        data["health_checks"] = {label: status for label, status in health_checks}
        print(json.dumps(data, indent=2))
    else:
        print_health_summary(health_checks)


if __name__ == "__main__":
    main()
