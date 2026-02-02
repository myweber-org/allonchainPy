
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organize files in the given directory by moving them into subdirectories
    based on their file extensions.
    """
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return
    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        if os.path.isfile(item_path):
            file_extension = Path(item).suffix.lower()
            
            if file_extension:
                target_dir = os.path.join(directory, file_extension[1:])
            else:
                target_dir = os.path.join(directory, "no_extension")
            
            os.makedirs(target_dir, exist_ok=True)
            
            try:
                shutil.move(item_path, os.path.join(target_dir, item))
                print(f"Moved: {item} -> {target_dir}")
            except Exception as e:
                print(f"Error moving {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)
    print("File organization complete.")
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organizes files in the specified directory by moving them into
    subfolders named after their file extensions.
    """
    if not os.path.isdir(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        return

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)

        if os.path.isfile(item_path):
            file_extension = Path(item).suffix.lower()

            if file_extension:
                folder_name = file_extension[1:] + "_files"
            else:
                folder_name = "no_extension_files"

            target_folder = os.path.join(directory, folder_name)
            os.makedirs(target_folder, exist_ok=True)

            try:
                shutil.move(item_path, os.path.join(target_folder, item))
                print(f"Moved: {item} -> {folder_name}/")
            except Exception as e:
                print(f"Failed to move {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil
from pathlib import Path

def organize_files(directory_path):
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory.")
        return

    categories = {
        'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'audio': ['.mp3', '.wav', '.flac', '.aac'],
        'video': ['.mp4', '.avi', '.mov', '.mkv'],
        'archives': ['.zip', '.tar', '.gz', '.rar'],
        'code': ['.py', '.js', '.html', '.css', '.java', '.cpp']
    }

    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isfile(item_path):
            file_ext = Path(item).suffix.lower()
            moved = False

            for category, extensions in categories.items():
                if file_ext in extensions:
                    category_dir = os.path.join(directory_path, category)
                    os.makedirs(category_dir, exist_ok=True)
                    dest_path = os.path.join(category_dir, item)
                    
                    counter = 1
                    while os.path.exists(dest_path):
                        name, ext = os.path.splitext(item)
                        dest_path = os.path.join(category_dir, f"{name}_{counter}{ext}")
                        counter += 1
                    
                    shutil.move(item_path, dest_path)
                    print(f"Moved: {item} -> {category}/")
                    moved = True
                    break

            if not moved:
                other_dir = os.path.join(directory_path, 'other')
                os.makedirs(other_dir, exist_ok=True)
                dest_path = os.path.join(other_dir, item)
                
                counter = 1
                while os.path.exists(dest_path):
                    name, ext = os.path.splitext(item)
                    dest_path = os.path.join(other_dir, f"{name}_{counter}{ext}")
                    counter += 1
                
                shutil.move(item_path, dest_path)
                print(f"Moved: {item} -> other/")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)