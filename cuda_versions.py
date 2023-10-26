import os
import subprocess


def find_cuda_versions():
    # Using 'find' command to search for cuda directories in typical installation paths
    cmd = "find /usr/local/ -maxdepth 1 -type d -name 'cuda*' 2>/dev/null"

    try:
        output = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
        if output:
            versions = [os.path.basename(path) for path in output.split("\n")]
            return versions
        else:
            return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def main():
    versions = find_cuda_versions()
    if versions:
        print("Found the following CUDA versions:")
        for version in versions:
            print(version)
    else:
        print("No CUDA versions found!")


if __name__ == "__main__":
    main()
