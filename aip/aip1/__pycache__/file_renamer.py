
import os
import glob
from pathlib import Path

def rename_files_sequentially(directory, prefix="file", extension=".txt"):
    files = list(Path(directory).glob(f"*{extension}"))
    
    files_sorted = sorted(files, key=lambda x: x.stat().st_ctime)
    
    for index, file_path in enumerate(files_sorted, start=1):
        new_name = f"{prefix}_{index:03d}{extension}"
        new_path = file_path.parent / new_name
        
        if new_path.exists():
            print(f"Warning: {new_path} already exists. Skipping rename.")
            continue
            
        try:
            file_path.rename(new_path)
            print(f"Renamed: {file_path.name} -> {new_name}")
        except Exception as e:
            print(f"Error renaming {file_path.name}: {e}")

if __name__ == "__main__":
    target_dir = input("Enter directory path: ").strip()
    
    if not os.path.isdir(target_dir):
        print("Invalid directory path.")
    else:
        rename_files_sequentially(target_dir)