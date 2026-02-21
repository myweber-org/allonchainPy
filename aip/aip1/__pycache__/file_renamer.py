
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
import os
import sys

def batch_rename(directory, prefix):
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        files.sort()
        
        for index, filename in enumerate(files, start=1):
            file_extension = os.path.splitext(filename)[1]
            new_name = f"{prefix}_{index:03d}{file_extension}"
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")
            
        print(f"Successfully renamed {len(files)} files.")
        
    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.")
    except PermissionError:
        print(f"Error: Permission denied for directory '{directory}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python file_renamer.py <directory_path> <prefix>")
        sys.exit(1)
    
    target_directory = sys.argv[1]
    name_prefix = sys.argv[2]
    batch_rename(target_directory, name_prefix)