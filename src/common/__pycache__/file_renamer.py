
import os
import sys
from pathlib import Path
from datetime import datetime

def rename_files_with_sequence(directory_path, prefix="file"):
    try:
        path = Path(directory_path)
        if not path.exists() or not path.is_dir():
            print(f"Error: {directory_path} is not a valid directory.")
            return False

        files = []
        for item in path.iterdir():
            if item.is_file():
                try:
                    creation_time = item.stat().st_ctime
                    files.append((creation_time, item))
                except OSError:
                    continue

        files.sort(key=lambda x: x[0])

        for index, (_, file_path) in enumerate(files, start=1):
            extension = file_path.suffix
            new_name = f"{prefix}_{index:03d}{extension}"
            new_path = file_path.parent / new_name

            try:
                file_path.rename(new_path)
                print(f"Renamed: {file_path.name} -> {new_name}")
            except OSError as e:
                print(f"Failed to rename {file_path.name}: {e}")

        return True

    except Exception as e:
        print(f"An error occurred: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory_path> [prefix]")
        sys.exit(1)

    target_directory = sys.argv[1]
    name_prefix = sys.argv[2] if len(sys.argv) > 2 else "file"

    success = rename_files_with_sequence(target_directory, name_prefix)
    sys.exit(0 if success else 1)
import os
import datetime

def rename_files_in_directory(directory_path, prefix="file"):
    try:
        files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        
        for filename in files:
            file_path = os.path.join(directory_path, filename)
            creation_time = os.path.getctime(file_path)
            date_str = datetime.datetime.fromtimestamp(creation_time).strftime('%Y%m%d_%H%M%S')
            
            file_extension = os.path.splitext(filename)[1]
            new_filename = f"{prefix}_{date_str}{file_extension}"
            new_file_path = os.path.join(directory_path, new_filename)
            
            counter = 1
            while os.path.exists(new_file_path):
                new_filename = f"{prefix}_{date_str}_{counter}{file_extension}"
                new_file_path = os.path.join(directory_path, new_filename)
                counter += 1
            
            os.rename(file_path, new_file_path)
            print(f"Renamed: {filename} -> {new_filename}")
            
        print(f"Successfully renamed {len(files)} files.")
        
    except FileNotFoundError:
        print(f"Directory not found: {directory_path}")
    except PermissionError:
        print(f"Permission denied for directory: {directory_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    target_directory = input("Enter directory path: ").strip()
    custom_prefix = input("Enter file prefix (press Enter for default 'file'): ").strip()
    
    if not custom_prefix:
        custom_prefix = "file"
    
    rename_files_in_directory(target_directory, custom_prefix)