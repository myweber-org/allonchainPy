
import os
import re
import sys

def rename_files(directory, pattern, replacement):
    """
    Rename files in the specified directory based on a regex pattern.
    
    Args:
        directory (str): Path to the directory containing files.
        pattern (str): Regex pattern to match in filenames.
        replacement (str): String to replace matched pattern.
    """
    try:
        if not os.path.isdir(directory):
            print(f"Error: Directory '{directory}' does not exist.")
            return False
        
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        renamed_count = 0
        
        for filename in files:
            new_name = re.sub(pattern, replacement, filename)
            if new_name != filename:
                old_path = os.path.join(directory, filename)
                new_path = os.path.join(directory, new_name)
                
                # Avoid overwriting existing files
                if os.path.exists(new_path):
                    print(f"Warning: Skipping '{filename}' -> '{new_name}' (target exists)")
                    continue
                
                os.rename(old_path, new_path)
                print(f"Renamed: '{filename}' -> '{new_name}'")
                renamed_count += 1
        
        print(f"\nRenaming complete. {renamed_count} file(s) renamed.")
        return True
        
    except Exception as e:
        print(f"Error during renaming: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python file_renamer.py <directory> <pattern> <replacement>")
        print("Example: python file_renamer.py ./photos 'IMG_\\d+' 'Vacation_'")
        sys.exit(1)
    
    dir_path = sys.argv[1]
    regex_pattern = sys.argv[2]
    replace_with = sys.argv[3]
    
    rename_files(dir_path, regex_pattern, replace_with)
import os
import sys
from datetime import datetime

def rename_files_by_date(directory):
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        
        for filename in files:
            filepath = os.path.join(directory, filename)
            stat = os.stat(filepath)
            creation_time = datetime.fromtimestamp(stat.st_ctime)
            
            new_name = creation_time.strftime("%Y%m%d_%H%M%S") + os.path.splitext(filename)[1]
            new_path = os.path.join(directory, new_name)
            
            counter = 1
            while os.path.exists(new_path):
                name_part = creation_time.strftime("%Y%m%d_%H%M%S")
                ext_part = os.path.splitext(filename)[1]
                new_name = f"{name_part}_{counter}{ext_part}"
                new_path = os.path.join(directory, new_name)
                counter += 1
            
            os.rename(filepath, new_path)
            print(f"Renamed: {filename} -> {new_name}")
            
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python file_renamer.py <directory_path>")
        sys.exit(1)
    
    target_dir = sys.argv[1]
    if not os.path.isdir(target_dir):
        print(f"Error: {target_dir} is not a valid directory")
        sys.exit(1)
    
    success = rename_files_by_date(target_dir)
    sys.exit(0 if success else 1)
import os
import sys

def rename_files_sequentially(directory, prefix="file", extension=".txt"):
    """
    Rename all files in the given directory with sequential numbering.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return False

    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    files.sort()

    for index, filename in enumerate(files, start=1):
        old_path = os.path.join(directory, filename)
        new_filename = f"{prefix}_{index:03d}{extension}"
        new_path = os.path.join(directory, new_filename)

        try:
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")
        except OSError as e:
            print(f"Failed to rename {filename}: {e}")

    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory> [prefix] [extension]")
        sys.exit(1)

    dir_path = sys.argv[1]
    prefix_arg = sys.argv[2] if len(sys.argv) > 2 else "file"
    extension_arg = sys.argv[3] if len(sys.argv) > 3 else ".txt"

    rename_files_sequentially(dir_path, prefix_arg, extension_arg)