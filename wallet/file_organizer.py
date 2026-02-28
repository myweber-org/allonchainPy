
import os
import shutil
from pathlib import Path

def organize_files(directory_path):
    """
    Organizes files in the specified directory by moving them into
    subfolders named after their file extensions.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: The path '{directory_path}' is not a valid directory.")
        return

    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)

        if os.path.isfile(item_path):
            file_extension = Path(item).suffix.lower()

            if file_extension:
                folder_name = file_extension[1:] + "_files"
            else:
                folder_name = "no_extension_files"

            target_folder = os.path.join(directory_path, folder_name)
            os.makedirs(target_folder, exist_ok=True)

            try:
                shutil.move(item_path, target_folder)
                print(f"Moved: {item} -> {folder_name}/")
            except Exception as e:
                print(f"Failed to move {item}: {e}")

if __name__ == "__main__":
    target_dir = input("Enter the directory path to organize: ").strip()
    organize_files(target_dir)
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organizes files in the given directory by moving them into subfolders
    based on their file extensions.
    """
    # Define file type categories
    categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Audio': ['.mp3', '.wav', '.aac', '.flac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv'],
        'Archives': ['.zip', '.tar', '.gz', '.rar'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp']
    }

    # Ensure the directory exists
    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"Error: Directory '{directory}' does not exist.")
        return

    # Create category folders if they don't exist
    for category in categories:
        (dir_path / category).mkdir(exist_ok=True)

    # Track moved files
    moved_files = []

    # Iterate over files in the directory
    for item in dir_path.iterdir():
        if item.is_file():
            file_ext = item.suffix.lower()
            moved = False

            # Find the appropriate category
            for category, extensions in categories.items():
                if file_ext in extensions:
                    dest_dir = dir_path / category
                    try:
                        shutil.move(str(item), str(dest_dir / item.name))
                        moved_files.append((item.name, category))
                        moved = True
                        break
                    except Exception as e:
                        print(f"Failed to move {item.name}: {e}")

            # If no category matched, move to 'Other'
            if not moved:
                other_dir = dir_path / 'Other'
                other_dir.mkdir(exist_ok=True)
                try:
                    shutil.move(str(item), str(other_dir / item.name))
                    moved_files.append((item.name, 'Other'))
                except Exception as e:
                    print(f"Failed to move {item.name}: {e}")

    # Print summary
    if moved_files:
        print(f"Organized {len(moved_files)} file(s):")
        for filename, category in moved_files:
            print(f"  {filename} -> {category}/")
    else:
        print("No files were moved.")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil

def organize_files(directory):
    """
    Organize files in the specified directory into folders based on their extensions.
    """
    # Define file type categories and their associated extensions
    file_categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Audio': ['.mp3', '.wav', '.aac', '.flac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv'],
        'Archives': ['.zip', '.tar', '.gz', '.rar', '.7z'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c'],
    }

    # Ensure the directory exists
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    # Get all files in the directory
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    except PermissionError:
        print(f"Error: Permission denied for directory '{directory}'.")
        return

    moved_count = 0

    for filename in files:
        file_path = os.path.join(directory, filename)
        # Get the file extension
        _, extension = os.path.splitext(filename)
        extension = extension.lower()

        # Find the category for the extension
        target_category = None
        for category, extensions in file_categories.items():
            if extension in extensions:
                target_category = category
                break

        # If no category found, put in 'Other'
        if target_category is None:
            target_category = 'Other'

        # Create target category folder if it doesn't exist
        target_folder = os.path.join(directory, target_category)
        os.makedirs(target_folder, exist_ok=True)

        # Move the file
        target_path = os.path.join(target_folder, filename)
        # Handle potential name conflicts
        if os.path.exists(target_path):
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(target_path):
                new_filename = f"{base}_{counter}{ext}"
                target_path = os.path.join(target_folder, new_filename)
                counter += 1

        try:
            shutil.move(file_path, target_path)
            moved_count += 1
            print(f"Moved: {filename} -> {target_category}/")
        except Exception as e:
            print(f"Failed to move {filename}: {e}")

    print(f"\nOrganization complete. Moved {moved_count} file(s).")

if __name__ == "__main__":
    # Example usage: organize files in the current directory
    current_dir = os.getcwd()
    organize_files(current_dir)
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
            
            if not file_extension:
                folder_name = "no_extension"
            else:
                folder_name = file_extension[1:] + "_files"
            
            target_folder = os.path.join(directory, folder_name)
            os.makedirs(target_folder, exist_ok=True)
            
            try:
                shutil.move(item_path, os.path.join(target_folder, item))
                print(f"Moved: {item} -> {folder_name}/")
            except Exception as e:
                print(f"Error moving {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)
    print("File organization complete.")