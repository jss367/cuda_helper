"""
It's possible, and it's a common setup when using Conda environments.

PATH: When you activate a Conda environment, Conda will prepend the bin directory of that environment to your PATH. This means that any executables installed in the Conda environment will be preferred over system-wide installations. This allows you to have different versions of tools or libraries in different Conda environments and to switch between them simply by activating and deactivating environments.

LD_LIBRARY_PATH: Conda might adjust this variable to ensure that libraries specific to the activated environment are found before system-wide libraries. This is less common than the adjustments to PATH, but it can happen, especially when dealing with packages that have specific library dependencies that might conflict with system-wide installations.
"""
import json
import os
import subprocess


def get_env_var_from_system(var_name):
    value = os.environ.get(var_name, "")
    print(f"[System] {var_name}: {value}")
    return value


def get_env_var_from_conda_env(env_name, var_name):
    command = f"conda activate {env_name} && echo ${{{var_name}}}"
    try:
        result = subprocess.check_output(command, shell=True, executable="/bin/bash")
        value = result.decode().strip()
        print(f"[{env_name}] {var_name}: {value}")
        return value
    except subprocess.CalledProcessError as e:
        print(f"Error executing command for environment {env_name}: {e}")
        return ""


def get_conda_envs():
    command = "conda env list --json"
    try:
        result = subprocess.check_output(command, shell=True)
        data = json.loads(result.decode())
        envs = data["envs"]
        env_names = [env.split("/")[-1] for env in envs]
        print(f"Found conda environments: {', '.join(env_names)}")
        return env_names
    except subprocess.CalledProcessError as e:
        print(f"Error executing command to list conda environments: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON data: {e}")
        return []


def main():
    env_vars = ["PATH", "LD_LIBRARY_PATH"]
    print("Fetching system-wide environment variables...")
    system_vals = {var: get_env_var_from_system(var) for var in env_vars}

    print("\nFetching conda environments...")
    conda_envs = get_conda_envs()

    for env in conda_envs:
        print(f"\nChecking Conda environment: {env}")
        for var in env_vars:
            conda_val = get_env_var_from_conda_env(env, var)
            if system_vals[var] != conda_val:
                print(f"\033[91mDifference detected in {var} for environment {env}\033[0m")  # Red text
            else:
                print(f"{var} in {env} matches system value.")
        print("\n")


if __name__ == "__main__":
    main()
