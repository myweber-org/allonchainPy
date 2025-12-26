
import os
import shutil
from pathlib import Path

def organize_files_by_extension(directory_path):
    """
    Organize files in the given directory by moving them into subdirectories
    named after their file extensions.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a valid directory.")
        return

    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        
        if os.path.isfile(item_path):
            file_extension = Path(item).suffix.lower()
            
            if file_extension:
                target_dir = os.path.join(directory_path, file_extension[1:])
            else:
                target_dir = os.path.join(directory_path, "no_extension")
            
            os.makedirs(target_dir, exist_ok=True)
            
            try:
                shutil.move(item_path, os.path.join(target_dir, item))
                print(f"Moved: {item} -> {target_dir}/")
            except Exception as e:
                print(f"Failed to move {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files_by_extension(target_directory)