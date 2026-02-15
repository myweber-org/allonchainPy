
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organize files in the specified directory by moving them into
    folders named after their file extensions.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
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
            
            target_path = os.path.join(target_folder, item)
            
            try:
                shutil.move(item_path, target_path)
                print(f"Moved: {item} -> {folder_name}/")
            except Exception as e:
                print(f"Failed to move {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)
    print("File organization complete.")
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organize files in the specified directory by moving them into
    subfolders based on their file extensions.
    """
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
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
            
            target_path = os.path.join(target_folder, item)
            
            if not os.path.exists(target_path):
                shutil.move(item_path, target_path)
                print(f"Moved: {item} -> {folder_name}/")
            else:
                print(f"Skipped: {item} already exists in {folder_name}/")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil
from pathlib import Path

def organize_files_by_extension(directory_path):
    """
    Organize files in the specified directory into subfolders based on their extensions.
    """
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return

    path = Path(directory_path)
    
    # Define categories and their associated extensions
    categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Archives': ['.zip', '.tar', '.gz', '.rar', '.7z'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp'],
        'Audio': ['.mp3', '.wav', '.flac', '.aac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    }
    
    # Create a mapping from extension to category
    extension_to_category = {}
    for category, extensions in categories.items():
        for ext in extensions:
            extension_to_category[ext] = category
    
    # Process each file in the directory
    for item in path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            category = extension_to_category.get(file_extension, 'Other')
            
            # Create category folder if it doesn't exist
            category_folder = path / category
            category_folder.mkdir(exist_ok=True)
            
            # Move file to category folder
            try:
                shutil.move(str(item), str(category_folder / item.name))
                print(f"Moved '{item.name}' to '{category}' folder.")
            except Exception as e:
                print(f"Error moving '{item.name}': {e}")
    
    print("File organization completed.")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files_by_extension(target_directory)