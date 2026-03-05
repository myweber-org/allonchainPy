
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organize files in the specified directory by moving them into
    subfolders based on their file extensions.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
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
    Organize files in the given directory by moving them into folders
    named after their file extensions.
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
            shutil.move(item_path, target_path)
            print(f"Moved: {item} -> {folder_name}/")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil

def organize_files(directory):
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            file_extension = filename.split('.')[-1]
            target_dir = os.path.join(directory, file_extension)
            
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            source_path = os.path.join(directory, filename)
            target_path = os.path.join(target_dir, filename)
            shutil.move(source_path, target_path)
            print(f"Moved {filename} to {file_extension}/")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ")
    if os.path.exists(target_directory):
        organize_files(target_directory)
        print("File organization complete.")
    else:
        print("Directory does not exist.")
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organizes files in the given directory by moving them into
    subfolders based on their file extensions.
    """
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory.")
        return

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)

        if os.path.isfile(item_path):
            file_extension = Path(item).suffix.lower()

            if not file_extension:
                folder_name = "NoExtension"
            else:
                folder_name = file_extension[1:].capitalize() + "Files"

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

def organize_files(directory):
    """
    Organizes files in the given directory by moving them into subfolders
    based on their file extensions.
    """
    # Define file type categories and their associated extensions
    file_categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp'],
        'Documents': ['.pdf', '.docx', '.txt', '.md', '.xlsx', '.pptx', '.csv'],
        'Archives': ['.zip', '.tar', '.gz', '.rar', '.7z'],
        'Audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg'],
        'Video': ['.mp4', '.avi', '.mkv', '.mov', '.wmv'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.json'],
        'Executables': ['.exe', '.msi', '.sh', '.bat', '.app']
    }
    
    # Convert directory to Path object for easier handling
    base_path = Path(directory)
    
    # Check if the directory exists
    if not base_path.exists() or not base_path.is_dir():
        print(f"Error: Directory '{directory}' does not exist or is not a directory.")
        return
    
    # Create category folders if they don't exist
    for category in file_categories.keys():
        category_path = base_path / category
        category_path.mkdir(exist_ok=True)
    
    # Track moved files and errors
    moved_files = []
    error_files = []
    
    # Iterate through all files in the directory
    for item in base_path.iterdir():
        # Skip directories
        if item.is_dir():
            continue
            
        # Get file extension
        file_extension = item.suffix.lower()
        
        # Find the appropriate category for the file
        target_category = None
        for category, extensions in file_categories.items():
            if file_extension in extensions:
                target_category = category
                break
        
        # If no category found, move to 'Other' folder
        if target_category is None:
            target_category = 'Other'
            other_path = base_path / target_category
            other_path.mkdir(exist_ok=True)
        
        # Construct target path
        target_path = base_path / target_category / item.name
        
        # Move the file
        try:
            # Handle naming conflicts
            if target_path.exists():
                # Add a number to the filename if it already exists
                counter = 1
                name_parts = item.stem, item.suffix
                while target_path.exists():
                    new_name = f"{name_parts[0]}_{counter}{name_parts[1]}"
                    target_path = base_path / target_category / new_name
                    counter += 1
            
            shutil.move(str(item), str(target_path))
            moved_files.append((item.name, target_category))
            
        except Exception as e:
            error_files.append((item.name, str(e)))
    
    # Print summary
    print(f"\nFile organization complete for: {directory}")
    print(f"Total files moved: {len(moved_files)}")
    
    if moved_files:
        print("\nMoved files:")
        for filename, category in moved_files:
            print(f"  {filename} -> {category}/")
    
    if error_files:
        print(f"\nErrors ({len(error_files)} files):")
        for filename, error in error_files:
            print(f"  {filename}: {error}")

if __name__ == "__main__":
    # Example usage: organize files in the current directory
    current_directory = os.getcwd()
    organize_files(current_directory)