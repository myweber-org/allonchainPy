
import os
import glob
from pathlib import Path
from datetime import datetime

def rename_files_sequentially(directory, prefix="file", extension=".txt"):
    files = glob.glob(os.path.join(directory, "*" + extension))
    files.sort(key=os.path.getctime)
    
    for index, file_path in enumerate(files, start=1):
        new_name = f"{prefix}_{index:03d}{extension}"
        new_path = os.path.join(directory, new_name)
        os.rename(file_path, new_path)
        print(f"Renamed: {Path(file_path).name} -> {new_name}")

if __name__ == "__main__":
    target_dir = "./documents"
    if os.path.exists(target_dir):
        rename_files_sequentially(target_dir, prefix="document", extension=".pdf")
    else:
        print(f"Directory '{target_dir}' does not exist.")
import os
import sys

def rename_files_with_sequential_numbers(directory, prefix="file"):
    """
    Rename all files in the specified directory with sequential numbering.
    Files are sorted alphabetically before renaming.
    """
    try:
        if not os.path.isdir(directory):
            print(f"Error: {directory} is not a valid directory.")
            return False
        
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        files.sort()
        
        for index, filename in enumerate(files, start=1):
            file_extension = os.path.splitext(filename)[1]
            new_filename = f"{prefix}_{index:03d}{file_extension}"
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            
            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")
            except OSError as e:
                print(f"Failed to rename {filename}: {e}")
        
        print(f"Successfully renamed {len(files)} files.")
        return True
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory> [prefix]")
        print("Example: python file_renamer.py ./photos vacation")
        sys.exit(1)
    
    target_directory = sys.argv[1]
    name_prefix = sys.argv[2] if len(sys.argv) > 2 else "file"
    
    rename_files_with_sequential_numbers(target_directory, name_prefix)