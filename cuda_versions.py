import os
import subprocess


def find_cuda_versions():
    # Using 'find' command to search for cuda directories in typical installation paths
    cmd = "find /usr/local/ -maxdepth 1 -type d -name 'cuda*' 2>/dev/null"

    try:
        output = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
        if output:
            paths = output.split("\n")
            return paths
        else:
            return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def print_red(text):
    # ANSI escape code to print in red
    RED = "\033[91m"
    RESET = "\033[0m"
    print(RED + text + RESET)


def main():
    paths = find_cuda_versions()
    if paths:
        print("Found the following CUDA versions:")
        for path in paths:
            print(f"{os.path.basename(path)} at {path}")

        # Check for multiple CUDA versions
        if len(paths) > 1:
            print_red("\nAlert: Multiple CUDA versions detected. This may cause conflicts.")
    else:
        print("No CUDA versions found!")


if __name__ == "__main__":
    main()
