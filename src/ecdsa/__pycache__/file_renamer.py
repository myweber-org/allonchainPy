
import os
import glob
from pathlib import Path
from datetime import datetime

def rename_files_sequentially(directory, prefix="file", extension=".txt"):
    files = glob.glob(os.path.join(directory, f"*{extension}"))
    files.sort(key=os.path.getctime)
    
    for index, filepath in enumerate(files, start=1):
        new_name = f"{prefix}_{index:03d}{extension}"
        new_path = os.path.join(directory, new_name)
        try:
            os.rename(filepath, new_path)
            print(f"Renamed: {Path(filepath).name} -> {new_name}")
        except OSError as e:
            print(f"Error renaming {filepath}: {e}")

if __name__ == "__main__":
    target_dir = input("Enter directory path: ").strip()
    if os.path.isdir(target_dir):
        rename_files_sequentially(target_dir)
    else:
        print("Invalid directory path")