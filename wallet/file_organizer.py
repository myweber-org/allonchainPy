
import os
import shutil
from pathlib import Path

def organize_files(directory_path):
    """
    Organizes files in the given directory by moving them into subfolders
    named after their file extensions.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: The path '{directory_path}' is not a valid directory.")
        return

    base_path = Path(directory_path)

    for item in base_path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            if not file_extension:
                folder_name = "no_extension"
            else:
                folder_name = file_extension[1:]

            target_folder = base_path / folder_name
            target_folder.mkdir(exist_ok=True)

            try:
                shutil.move(str(item), str(target_folder / item.name))
                print(f"Moved: {item.name} -> {folder_name}/")
            except Exception as e:
                print(f"Failed to move {item.name}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil
from pathlib import Path

def organize_files(directory_path):
    """
    Organizes files in the given directory by moving them into
    subfolders based on their file extensions.
    """
    # Define file type categories and their associated extensions
    file_categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Archives': ['.zip', '.tar', '.gz', '.rar', '.7z'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp'],
        'Audio': ['.mp3', '.wav', '.flac', '.aac'],
        'Video': ['.mp4', '.avi', '.mkv', '.mov']
    }

    # Ensure the directory path exists
    path = Path(directory_path)
    if not path.exists() or not path.is_dir():
        print(f"Error: The directory '{directory_path}' does not exist.")
        return

    # Create category folders if they don't exist
    for category in file_categories.keys():
        category_path = path / category
        category_path.mkdir(exist_ok=True)

    # Track moved files and extensions not categorized
    moved_files = []
    uncategorized_extensions = set()

    # Iterate over all items in the directory
    for item in path.iterdir():
        # Skip directories and hidden files
        if item.is_dir() or item.name.startswith('.'):
            continue

        # Get file extension
        file_extension = item.suffix.lower()

        # Find the appropriate category for the file
        target_category = None
        for category, extensions in file_categories.items():
            if file_extension in extensions:
                target_category = category
                break

        # Move file to the appropriate category folder
        if target_category:
            target_path = path / target_category / item.name
            # Handle duplicate file names
            if target_path.exists():
                base_name = item.stem
                counter = 1
                while target_path.exists():
                    new_name = f"{base_name}_{counter}{item.suffix}"
                    target_path = path / target_category / new_name
                    counter += 1

            try:
                shutil.move(str(item), str(target_path))
                moved_files.append((item.name, target_category))
            except Exception as e:
                print(f"Error moving file {item.name}: {e}")
        else:
            uncategorized_extensions.add(file_extension)

    # Print summary
    if moved_files:
        print(f"Successfully organized {len(moved_files)} file(s):")
        for file_name, category in moved_files:
            print(f"  - {file_name} -> {category}/")
    else:
        print("No files were moved.")

    if uncategorized_extensions:
        print(f"\nUncategorized file extensions found: {', '.join(sorted(uncategorized_extensions))}")

if __name__ == "__main__":
    # Example usage: organize files in the current directory
    target_directory = input("Enter the directory path to organize (or press Enter for current directory): ").strip()
    
    if not target_directory:
        target_directory = os.getcwd()
    
    organize_files(target_directory)