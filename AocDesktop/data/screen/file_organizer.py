
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organizes files in the given directory by moving them into
    subfolders based on their file extensions.
    """
    # Define categories and their associated file extensions
    categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Archives': ['.zip', '.tar', '.gz', '.rar', '.7z'],
        'Code': ['.py', '.js', '.html', '.css', '.json', '.xml'],
        'Audio': ['.mp3', '.wav', '.flac', '.aac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    }

    # Create a reverse lookup dictionary for quick extension to category mapping
    extension_to_category = {}
    for category, extensions in categories.items():
        for ext in extensions:
            extension_to_category[ext.lower()] = category

    # Ensure the directory exists
    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"Error: Directory '{directory}' does not exist.")
        return

    # Iterate over all items in the directory
    for item in dir_path.iterdir():
        if item.is_file():
            # Get the file extension
            ext = item.suffix.lower()
            # Determine the category
            category = extension_to_category.get(ext, 'Other')
            # Create the category folder if it doesn't exist
            category_folder = dir_path / category
            category_folder.mkdir(exist_ok=True)
            # Move the file
            try:
                shutil.move(str(item), str(category_folder / item.name))
                print(f"Moved: {item.name} -> {category}/")
            except Exception as e:
                print(f"Failed to move {item.name}: {e}")

if __name__ == "__main__":
    # Use the current working directory as the target
    target_dir = os.getcwd()
    print(f"Organizing files in: {target_dir}")
    organize_files(target_dir)
    print("Organization complete.")