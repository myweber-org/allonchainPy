
import os
import shutil
from pathlib import Path

def organize_files(directory):
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            file_extension = Path(item).suffix.lower()
            if file_extension:
                target_folder = os.path.join(directory, file_extension[1:] + "_files")
            else:
                target_folder = os.path.join(directory, "no_extension_files")
            
            os.makedirs(target_folder, exist_ok=True)
            shutil.move(item_path, os.path.join(target_folder, item))
            print(f"Moved: {item} -> {target_folder}")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)