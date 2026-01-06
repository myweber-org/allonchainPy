
import os
import shutil

def organize_files(directory):
    """
    Organize files in the given directory by moving them into folders
    named after their file extensions.
    """
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path):
            file_ext = filename.split('.')[-1] if '.' in filename else 'no_extension'
            target_dir = os.path.join(directory, file_ext)

            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            try:
                shutil.move(file_path, os.path.join(target_dir, filename))
                print(f"Moved '{filename}' to '{file_ext}/'")
            except Exception as e:
                print(f"Error moving '{filename}': {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)