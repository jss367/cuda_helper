import os
import subprocess


def find_cudnn_versions_unix():
    """
    This is unix only
    """
    # Using 'find' command to search for cudnn.h files throughout the system
    cmd_find_cudnn = "find /usr/ -type f -name 'cudnn.h' 2>/dev/null"

    try:
        paths = subprocess.check_output(cmd_find_cudnn, shell=True).decode("utf-8").strip().split("\n")
        versions = {}

        for path in paths:
            cmd_extract_version = f"grep 'define CUDNN_MAJOR' {path} -A 2"
            output = subprocess.check_output(cmd_extract_version, shell=True).decode("utf-8").strip().split("\n")

            if len(output) >= 3:
                major = output[0].split()[-1]
                minor = output[1].split()[-1]
                patch = output[2].split()[-1]

                version = f"{major}.{minor}.{patch}"
                versions[path] = version

        return versions

    except Exception as e:
        print(f"Error: {str(e)}")
        return {}


def main():
    versions = find_cudnn_versions_unix()
    if versions:
        print("Found the following cuDNN versions:")
        for path, version in versions.items():
            print(f"{version} at {path}")
    else:
        print("No cuDNN versions found!")


if __name__ == "__main__":
    main()
