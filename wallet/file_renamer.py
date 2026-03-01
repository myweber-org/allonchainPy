
import os
import re
import sys

def rename_files(directory, pattern, replacement):
    try:
        if not os.path.isdir(directory):
            print(f"Error: {directory} is not a valid directory.")
            return False

        files = os.listdir(directory)
        renamed_count = 0

        for filename in files:
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                new_name = re.sub(pattern, replacement, filename)
                if new_name != filename:
                    new_path = os.path.join(directory, new_name)
                    try:
                        os.rename(file_path, new_path)
                        print(f"Renamed: {filename} -> {new_name}")
                        renamed_count += 1
                    except OSError as e:
                        print(f"Failed to rename {filename}: {e}")

        print(f"Renaming complete. {renamed_count} files renamed.")
        return True

    except Exception as e:
        print(f"An error occurred: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python file_renamer.py <directory> <pattern> <replacement>")
        sys.exit(1)

    dir_path = sys.argv[1]
    regex_pattern = sys.argv[2]
    replace_with = sys.argv[3]

    rename_files(dir_path, regex_pattern, replace_with)
import os
import sys

def rename_files_with_sequence(directory, prefix="file", extension=".txt"):
    """
    Rename all files in the specified directory with sequential numbering.
    
    Args:
        directory (str): Path to the directory containing files to rename.
        prefix (str): Prefix for the new filenames.
        extension (str): File extension to filter and apply.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        sys.exit(1)
    
    files = [f for f in os.listdir(directory) 
             if os.path.isfile(os.path.join(directory, f)) and f.endswith(extension)]
    files.sort()
    
    if not files:
        print(f"No files with extension '{extension}' found in '{directory}'.")
        return
    
    for index, filename in enumerate(files, start=1):
        old_path = os.path.join(directory, filename)
        new_name = f"{prefix}_{index:03d}{extension}"
        new_path = os.path.join(directory, new_name)
        
        try:
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")
        except OSError as e:
            print(f"Failed to rename {filename}: {e}")
    
    print(f"Renaming complete. Processed {len(files)} files.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory> [prefix] [extension]")
        sys.exit(1)
    
    target_dir = sys.argv[1]
    name_prefix = sys.argv[2] if len(sys.argv) > 2 else "file"
    file_extension = sys.argv[3] if len(sys.argv) > 3 else ".txt"
    
    rename_files_with_sequence(target_dir, name_prefix, file_extension)