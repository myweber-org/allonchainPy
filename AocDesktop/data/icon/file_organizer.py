
import os
import shutil

def organize_files(directory):
    """
    Organize files in the given directory by moving them into subfolders
    based on their file extensions.
    """
    # Define categories and their associated file extensions
    categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Audio': ['.mp3', '.wav', '.aac', '.flac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv'],
        'Archives': ['.zip', '.tar', '.gz', '.rar'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp']
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

    # Create category folders and move files
    for file in files:
        file_path = os.path.join(directory, file)
        file_ext = os.path.splitext(file)[1].lower()

        # Find the category for the file extension
        target_category = 'Other'  # Default category
        for category, extensions in categories.items():
            if file_ext in extensions:
                target_category = category
                break

        # Create target folder if it doesn't exist
        target_folder = os.path.join(directory, target_category)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        # Move the file to the target folder
        try:
            shutil.move(file_path, os.path.join(target_folder, file))
            print(f"Moved: {file} -> {target_category}/")
        except shutil.Error as e:
            print(f"Error moving {file}: {e}")
        except PermissionError:
            print(f"Permission denied when moving {file}.")

    print("File organization complete.")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)