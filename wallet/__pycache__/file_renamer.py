
import os
import sys
from pathlib import Path
from datetime import datetime

def rename_files_sequentially(directory_path, prefix="file"):
    try:
        path = Path(directory_path)
        if not path.exists() or not path.is_dir():
            print(f"Error: {directory_path} is not a valid directory.")
            return False

        files = []
        for item in path.iterdir():
            if item.is_file():
                try:
                    creation_time = item.stat().st_ctime
                    files.append((creation_time, item))
                except OSError:
                    continue

        files.sort(key=lambda x: x[0])

        for index, (_, file_path) in enumerate(files, start=1):
            extension = file_path.suffix
            new_name = f"{prefix}_{index:03d}{extension}"
            new_path = file_path.parent / new_name

            try:
                file_path.rename(new_path)
                print(f"Renamed: {file_path.name} -> {new_name}")
            except OSError as e:
                print(f"Failed to rename {file_path.name}: {e}")

        return True

    except Exception as e:
        print(f"An error occurred: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory_path> [prefix]")
        sys.exit(1)

    dir_path = sys.argv[1]
    name_prefix = sys.argv[2] if len(sys.argv) > 2 else "file"
    rename_files_sequentially(dir_path, name_prefix)