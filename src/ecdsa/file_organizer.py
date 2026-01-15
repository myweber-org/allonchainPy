
import os
import shutil
from pathlib import Path

def organize_files(directory="."):
    """
    Organizes files in the given directory by moving them into folders
    named after their file extensions.
    """
    base_path = Path(directory).resolve()
    
    # Define categories and their associated extensions
    categories = {
        "Images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg"],
        "Documents": [".pdf", ".docx", ".txt", ".xlsx", ".pptx", ".md"],
        "Archives": [".zip", ".tar", ".gz", ".rar", ".7z"],
        "Code": [".py", ".js", ".html", ".css", ".java", ".cpp"],
        "Audio": [".mp3", ".wav", ".flac", ".aac"],
        "Video": [".mp4", ".avi", ".mov", ".mkv"]
    }
    
    # Create a dictionary for quick extension lookup
    extension_to_category = {}
    for category, extensions in categories.items():
        for ext in extensions:
            extension_to_category[ext] = category
    
    # Process each file in the directory
    for item in base_path.iterdir():
        if item.is_file():
            file_ext = item.suffix.lower()
            
            # Determine category
            category = extension_to_category.get(file_ext, "Other")
            
            # Create category folder if it doesn't exist
            category_folder = base_path / category
            category_folder.mkdir(exist_ok=True)
            
            # Move file to category folder
            try:
                destination = category_folder / item.name
                shutil.move(str(item), str(destination))
                print(f"Moved: {item.name} -> {category}/")
            except Exception as e:
                print(f"Error moving {item.name}: {e}")
    
    print("File organization complete.")

if __name__ == "__main__":
    # Get directory from user or use current directory
    target_dir = input("Enter directory to organize (press Enter for current): ").strip()
    if not target_dir:
        target_dir = "."
    
    organize_files(target_dir)
import os
import shutil

def organize_files(directory):
    """
    Organize files in the given directory by moving them into folders
    named after their file extensions.
    """
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory.")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path):
            file_ext = os.path.splitext(filename)[1].lower()

            if file_ext:
                target_dir = os.path.join(directory, file_ext[1:])
            else:
                target_dir = os.path.join(directory, "no_extension")

            os.makedirs(target_dir, exist_ok=True)
            shutil.move(file_path, os.path.join(target_dir, filename))
            print(f"Moved '{filename}' to '{target_dir}'")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)