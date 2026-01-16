
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
import os
import shutil
from pathlib import Path

def organize_files_by_extension(directory_path):
    """
    Organize files in the given directory by moving them into subfolders
    based on their file extensions.
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
        'Video': ['.mp4', '.avi', '.mov', '.mkv']
    }
    
    # Create category folders if they don't exist
    for category in categories:
        category_path = path / category
        category_path.mkdir(exist_ok=True)
    
    # Process each file in the directory
    for item in path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            moved = False
            
            # Find the appropriate category for the file
            for category, extensions in categories.items():
                if file_extension in extensions:
                    destination = path / category / item.name
                    
                    # Handle duplicate filenames
                    counter = 1
                    while destination.exists():
                        stem = item.stem
                        new_name = f"{stem}_{counter}{item.suffix}"
                        destination = path / category / new_name
                        counter += 1
                    
                    try:
                        shutil.move(str(item), str(destination))
                        print(f"Moved: {item.name} -> {category}/")
                        moved = True
                        break
                    except Exception as e:
                        print(f"Error moving {item.name}: {e}")
            
            # If file doesn't match any category, move to 'Other'
            if not moved:
                other_folder = path / 'Other'
                other_folder.mkdir(exist_ok=True)
                destination = other_folder / item.name
                
                # Handle duplicate filenames in Other folder
                counter = 1
                while destination.exists():
                    stem = item.stem
                    new_name = f"{stem}_{counter}{item.suffix}"
                    destination = other_folder / new_name
                    counter += 1
                
                try:
                    shutil.move(str(item), str(destination))
                    print(f"Moved: {item.name} -> Other/")
                except Exception as e:
                    print(f"Error moving {item.name}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files_by_extension(target_directory)
    print("File organization completed.")