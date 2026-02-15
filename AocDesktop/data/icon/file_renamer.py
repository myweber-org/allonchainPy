
import os
import sys
from datetime import datetime

def rename_files_with_timestamp(directory):
    """
    Rename all files in the given directory by adding a timestamp prefix.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
    renamed_count = 0

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        if os.path.isfile(filepath):
            new_filename = timestamp + filename
            new_filepath = os.path.join(directory, new_filename)
            
            try:
                os.rename(filepath, new_filepath)
                print(f"Renamed: {filename} -> {new_filename}")
                renamed_count += 1
            except OSError as e:
                print(f"Failed to rename {filename}: {e}")

    print(f"Renamed {renamed_count} file(s).")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python file_renamer.py <directory_path>")
        sys.exit(1)
    
    target_directory = sys.argv[1]
    rename_files_with_timestamp(target_directory)
import os
import sys

def rename_files_with_sequence(directory, prefix="file", extension=".txt"):
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        files.sort()
        
        for index, filename in enumerate(files, start=1):
            old_path = os.path.join(directory, filename)
            new_name = f"{prefix}_{index:03d}{extension}"
            new_path = os.path.join(directory, new_name)
            
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")
            
        print(f"Successfully renamed {len(files)} files.")
        
    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.")
    except PermissionError:
        print(f"Error: Permission denied for directory '{directory}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_directory = sys.argv[1]
        rename_files_with_sequence(target_directory)
    else:
        print("Usage: python file_renamer.py <directory_path>")