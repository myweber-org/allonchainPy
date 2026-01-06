
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
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    # Create category folders if they don't exist
    for category in file_categories.keys():
        category_path = os.path.join(directory, category)
        if not os.path.exists(category_path):
            os.makedirs(category_path)

    # Track files that don't match any category
    other_files = []

    # Iterate through all files in the directory
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)

        # Skip directories
        if os.path.isdir(item_path):
            continue

        # Get file extension
        file_extension = Path(item).suffix.lower()

        # Find the appropriate category for the file
        moved = False
        for category, extensions in file_categories.items():
            if file_extension in extensions:
                dest_folder = os.path.join(directory, category)
                shutil.move(item_path, os.path.join(dest_folder, item))
                print(f"Moved '{item}' to '{category}' folder.")
                moved = True
                break

        # If no category matched, add to other files list
        if not moved:
            other_files.append(item)

    # Handle files that didn't match any category
    if other_files:
        other_folder = os.path.join(directory, 'Other')
        if not os.path.exists(other_folder):
            os.makedirs(other_folder)

        for file_name in other_files:
            src_path = os.path.join(directory, file_name)
            shutil.move(src_path, os.path.join(other_folder, file_name))
            print(f"Moved '{file_name}' to 'Other' folder.")

    print("File organization completed.")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)