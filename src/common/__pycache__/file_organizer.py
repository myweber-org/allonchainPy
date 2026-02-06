
import os
import shutil
from pathlib import Path

def organize_files(directory="."):
    """
    Organize files in the specified directory by moving them into
    subfolders named after their file extensions.
    """
    base_path = Path(directory).resolve()

    # Define common categories and their associated extensions
    categories = {
        "Images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp"],
        "Documents": [".pdf", ".docx", ".txt", ".md", ".rtf", ".odt", ".xlsx", ".pptx"],
        "Archives": [".zip", ".tar", ".gz", ".rar", ".7z"],
        "Audio": [".mp3", ".wav", ".flac", ".aac", ".ogg"],
        "Video": [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv"],
        "Code": [".py", ".js", ".html", ".css", ".java", ".cpp", ".c", ".json", ".xml"],
        "Executables": [".exe", ".msi", ".sh", ".bat", ".app"],
    }

    # Create a mapping from extension to category folder
    ext_to_category = {}
    for category, extensions in categories.items():
        for ext in extensions:
            ext_to_category[ext.lower()] = category

    # Ensure category directories exist
    for category in categories.keys():
        (base_path / category).mkdir(exist_ok=True)

    # Track files moved and errors
    moved_files = []
    error_files = []

    for item in base_path.iterdir():
        # Skip directories and hidden files
        if item.is_dir() or item.name.startswith('.'):
            continue

        ext = item.suffix.lower()
        if ext in ext_to_category:
            target_dir = base_path / ext_to_category[ext]
            try:
                shutil.move(str(item), str(target_dir / item.name))
                moved_files.append((item.name, ext_to_category[ext]))
            except Exception as e:
                error_files.append((item.name, str(e)))
        else:
            # For uncategorized extensions, put in "Other"
            other_dir = base_path / "Other"
            other_dir.mkdir(exist_ok=True)
            try:
                shutil.move(str(item), str(other_dir / item.name))
                moved_files.append((item.name, "Other"))
            except Exception as e:
                error_files.append((item.name, str(e)))

    # Print summary
    print(f"Organization of '{base_path}' completed.")
    if moved_files:
        print(f"\nMoved {len(moved_files)} file(s):")
        for filename, category in moved_files:
            print(f"  {filename} -> {category}/")
    if error_files:
        print(f"\nEncountered {len(error_files)} error(s):")
        for filename, error_msg in error_files:
            print(f"  {filename}: {error_msg}")

if __name__ == "__main__":
    # Optional: specify a directory via command line argument
    import sys
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    organize_files(target_dir)
import os
import shutil

def organize_files(directory):
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if os.path.isfile(file_path):
            file_extension = filename.split('.')[-1] if '.' in filename else 'no_extension'
            target_dir = os.path.join(directory, file_extension)
            
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            shutil.move(file_path, os.path.join(target_dir, filename))
            print(f"Moved {filename} to {file_extension}/")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ")
    organize_files(target_directory)
import os
import shutil

def organize_files(directory):
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if os.path.isfile(file_path):
            _, extension = os.path.splitext(filename)
            extension = extension.lower()
            
            if extension:
                folder_name = extension[1:] + "_files"
            else:
                folder_name = "no_extension_files"
            
            target_folder = os.path.join(directory, folder_name)
            os.makedirs(target_folder, exist_ok=True)
            
            target_path = os.path.join(target_folder, filename)
            shutil.move(file_path, target_path)
            print(f"Moved: {filename} -> {folder_name}/")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)