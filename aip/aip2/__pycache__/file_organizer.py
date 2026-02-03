
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organizes files in the specified directory by moving them into
    subfolders based on their file extensions.
    """
    # Define file type categories and their associated extensions
    file_categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Audio': ['.mp3', '.wav', '.flac', '.aac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv'],
        'Archives': ['.zip', '.rar', '.tar', '.gz'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp'],
    }

    # Ensure the directory exists
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    # Get all files in the directory
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)

        # Skip if it's a directory
        if os.path.isdir(item_path):
            continue

        # Get file extension
        file_extension = Path(item).suffix.lower()

        # Find the appropriate category
        target_category = 'Other'  # Default category
        for category, extensions in file_categories.items():
            if file_extension in extensions:
                target_category = category
                break

        # Create target directory if it doesn't exist
        target_dir = os.path.join(directory, target_category)
        os.makedirs(target_dir, exist_ok=True)

        # Move the file
        try:
            shutil.move(item_path, os.path.join(target_dir, item))
            print(f"Moved: {item} -> {target_category}/")
        except Exception as e:
            print(f"Failed to move {item}: {e}")

    print("File organization complete.")

if __name__ == "__main__":
    # Get the current working directory as the target
    target_directory = os.getcwd()
    organize_files(target_directory)