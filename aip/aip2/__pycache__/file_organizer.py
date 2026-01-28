
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organizes files in the specified directory by moving them into
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
            
            try:
                shutil.move(item_path, os.path.join(target_folder, item))
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

def organize_files(directory="."):
    """
    Organizes files in the specified directory by moving them into
    subdirectories based on their file extensions.
    """
    base_path = Path(directory).resolve()
    
    # Define categories and their associated file extensions
    categories = {
        "Images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg"],
        "Documents": [".pdf", ".docx", ".txt", ".xlsx", ".pptx", ".md"],
        "Archives": [".zip", ".tar", ".gz", ".rar", ".7z"],
        "Code": [".py", ".js", ".html", ".css", ".java", ".cpp", ".c"],
        "Audio": [".mp3", ".wav", ".flac", ".aac"],
        "Video": [".mp4", ".avi", ".mov", ".mkv", ".flv"]
    }
    
    # Create category folders if they don't exist
    for category in categories:
        category_path = base_path / category
        category_path.mkdir(exist_ok=True)
    
    # Track moved files and errors
    moved_files = []
    errors = []
    
    # Iterate through all items in the directory
    for item in base_path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            moved = False
            
            # Find the appropriate category for the file
            for category, extensions in categories.items():
                if file_extension in extensions:
                    try:
                        destination = base_path / category / item.name
                        # Handle naming conflicts
                        if destination.exists():
                            counter = 1
                            while destination.exists():
                                new_name = f"{item.stem}_{counter}{item.suffix}"
                                destination = base_path / category / new_name
                                counter += 1
                        
                        shutil.move(str(item), str(destination))
                        moved_files.append((item.name, category))
                        moved = True
                        break
                    except Exception as e:
                        errors.append((item.name, str(e)))
            
            # If file doesn't match any category, move to "Other"
            if not moved:
                other_folder = base_path / "Other"
                other_folder.mkdir(exist_ok=True)
                try:
                    destination = other_folder / item.name
                    if destination.exists():
                        counter = 1
                        while destination.exists():
                            new_name = f"{item.stem}_{counter}{item.suffix}"
                            destination = other_folder / new_name
                            counter += 1
                    
                    shutil.move(str(item), str(destination))
                    moved_files.append((item.name, "Other"))
                except Exception as e:
                    errors.append((item.name, str(e)))
    
    # Print summary
    if moved_files:
        print(f"Successfully organized {len(moved_files)} file(s):")
        for filename, category in moved_files:
            print(f"  {filename} -> {category}/")
    
    if errors:
        print(f"\nEncountered {len(errors)} error(s):")
        for filename, error_msg in errors:
            print(f"  {filename}: {error_msg}")
    
    return moved_files, errors

if __name__ == "__main__":
    import sys
    
    # Use command line argument if provided, otherwise use current directory
    target_directory = sys.argv[1] if len(sys.argv) > 1 else "."
    
    print(f"Organizing files in: {target_directory}")
    moved, errors = organize_files(target_directory)
    
    if not moved and not errors:
        print("No files found to organize.")