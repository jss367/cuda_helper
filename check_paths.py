"""
It's possible, and it's a common setup when using Conda environments.

PATH: When you activate a Conda environment, Conda will prepend the bin directory of that environment to your PATH. This means that any executables installed in the Conda environment will be preferred over system-wide installations. This allows you to have different versions of tools or libraries in different Conda environments and to switch between them simply by activating and deactivating environments.

LD_LIBRARY_PATH: Conda might adjust this variable to ensure that libraries specific to the activated environment are found before system-wide libraries. This is less common than the adjustments to PATH, but it can happen, especially when dealing with packages that have specific library dependencies that might conflict with system-wide installations.
"""
import os
import subprocess


def get_env_var_from_system(var_name):
    return os.environ.get(var_name, "")


def get_env_var_from_conda_env(env_name, var_name):
    command = f"conda activate {env_name} && echo ${{{var_name}}}"
    result = subprocess.check_output(command, shell=True, executable="/bin/bash")
    return result.decode().strip()


def get_conda_envs():
    command = "conda env list --json"
    result = subprocess.check_output(command, shell=True)
    envs = result.decode().split("\n")
    env_list = []
    for line in envs:
        if "name" in line:
            env_name = line.split(":")[1].strip().replace('"', "")
            env_list.append(env_name)
    return env_list


def main():
    env_vars = ["PATH", "LD_LIBRARY_PATH"]
    system_vals = {var: get_env_var_from_system(var) for var in env_vars}

    conda_envs = get_conda_envs()

    for env in conda_envs:
        print(f"Checking Conda environment: {env}")
        for var in env_vars:
            conda_val = get_env_var_from_conda_env(env, var)
            if system_vals[var] != conda_val:
                print(f"\033[91m{var} in {env}: {conda_val}\033[0m")  # Red text
            else:
                print(f"{var} in {env}: {conda_val}")
        print("\n")


if __name__ == "__main__":
    main()
