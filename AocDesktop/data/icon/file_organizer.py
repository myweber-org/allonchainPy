
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
import os
import shutil
from pathlib import Path

def organize_files(directory):
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return

    extensions_folders = {
        'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'audio': ['.mp3', '.wav', '.flac', '.aac'],
        'video': ['.mp4', '.mkv', '.avi', '.mov'],
        'archives': ['.zip', '.tar', '.gz', '.rar', '.7z'],
        'code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c']
    }

    for folder in extensions_folders.keys():
        folder_path = os.path.join(directory, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            file_extension = Path(item).suffix.lower()
            moved = False

            for folder, extensions in extensions_folders.items():
                if file_extension in extensions:
                    dest_folder = os.path.join(directory, folder)
                    try:
                        shutil.move(item_path, os.path.join(dest_folder, item))
                        print(f"Moved {item} to {folder}/")
                        moved = True
                        break
                    except Exception as e:
                        print(f"Error moving {item}: {e}")

            if not moved:
                other_folder = os.path.join(directory, 'other')
                if not os.path.exists(other_folder):
                    os.makedirs(other_folder)
                try:
                    shutil.move(item_path, os.path.join(other_folder, item))
                    print(f"Moved {item} to other/")
                except Exception as e:
                    print(f"Error moving {item} to other: {e}")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)