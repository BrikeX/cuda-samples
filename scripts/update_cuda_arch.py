#!/usr/bin/env python3

import argparse
import os
import re

parser = argparse.ArgumentParser(description="Update CUDA architectures")
parser.add_argument(
    "--cuda-arch",
    type=int,
    default="89",
    help="CUDA architecture to update",
)
args = parser.parse_args()


def find_and_replace_cuda_architectures():
    # Root directory to start the search
    root_dir = "/workspace"

    files_found = 0
    files_modified = 0

    print(f"Searching for CMakeLists.txt files in {root_dir}...")

    # Walk through directory tree
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == "CMakeLists.txt":
                filepath = os.path.join(dirpath, filename)
                files_found += 1

                # Read the file
                with open(filepath, "r") as file:
                    content = file.read()

                # Find and replace the CMAKE_CUDA_ARCHITECTURES line
                pattern = r"set\s*\(\s*CMAKE_CUDA_ARCHITECTURES\s+[^)]+\)"
                new_content = re.sub(
                    pattern, f"set(CMAKE_CUDA_ARCHITECTURES {args.cuda_arch})", content
                )

                # If content was changed, write it back
                if new_content != content:
                    with open(filepath, "w") as file:
                        file.write(new_content)
                    files_modified += 1
                    print(f"Modified: {filepath}")

    print(f"\nSummary:")
    print(f"- Total CMakeLists.txt files found: {files_found}")
    print(f"- Files modified: {files_modified}")


if __name__ == "__main__":
    find_and_replace_cuda_architectures()
