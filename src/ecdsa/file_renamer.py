
import os
import sys

def batch_rename(directory, prefix):
    """
    Rename all files in the specified directory with a given prefix and sequential numbers.
    """
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        files.sort()
        
        for index, filename in enumerate(files, start=1):
            file_extension = os.path.splitext(filename)[1]
            new_name = f"{prefix}_{index:03d}{file_extension}"
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")
        
        print(f"Successfully renamed {len(files)} files.")
    
    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.")
        sys.exit(1)
    except PermissionError:
        print(f"Error: Permission denied for directory '{directory}'.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python file_renamer.py <directory_path> <prefix>")
        sys.exit(1)
    
    target_directory = sys.argv[1]
    name_prefix = sys.argv[2]
    
    batch_rename(target_directory, name_prefix)
import os
import datetime

def rename_files_by_date(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    for filename in files:
        filepath = os.path.join(directory, filename)
        creation_time = os.path.getctime(filepath)
        date_str = datetime.datetime.fromtimestamp(creation_time).strftime('%Y%m%d_%H%M%S')
        
        name, ext = os.path.splitext(filename)
        new_filename = f"{date_str}{ext}"
        new_filepath = os.path.join(directory, new_filename)
        
        counter = 1
        while os.path.exists(new_filepath):
            new_filename = f"{date_str}_{counter}{ext}"
            new_filepath = os.path.join(directory, new_filename)
            counter += 1
        
        os.rename(filepath, new_filepath)
        print(f"Renamed: {filename} -> {new_filename}")

if __name__ == "__main__":
    target_directory = input("Enter directory path: ")
    if os.path.isdir(target_directory):
        rename_files_by_date(target_directory)
    else:
        print("Invalid directory path")