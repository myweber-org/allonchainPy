import os
import re
import sys

def rename_files(directory, pattern, replacement):
    """
    Rename files in the specified directory based on a regex pattern.
    
    Args:
        directory (str): Path to the directory containing files to rename.
        pattern (str): Regex pattern to match in filenames.
        replacement (str): String to replace matched pattern with.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        sys.exit(1)
    
    try:
        regex = re.compile(pattern)
    except re.error as e:
        print(f"Error: Invalid regex pattern '{pattern}': {e}")
        sys.exit(1)
    
    renamed_count = 0
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        if os.path.isfile(filepath):
            new_filename = regex.sub(replacement, filename)
            
            if new_filename != filename:
                new_filepath = os.path.join(directory, new_filename)
                
                if os.path.exists(new_filepath):
                    print(f"Warning: Cannot rename '{filename}' to '{new_filename}' - target already exists.")
                    continue
                
                try:
                    os.rename(filepath, new_filepath)
                    print(f"Renamed: '{filename}' -> '{new_filename}'")
                    renamed_count += 1
                except OSError as e:
                    print(f"Error renaming '{filename}' to '{new_filename}': {e}")
    
    print(f"\nRenaming complete. {renamed_count} files renamed.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python file_renamer.py <directory> <pattern> <replacement>")
        print("Example: python file_renamer.py ./photos 'IMG_\\d+' 'Vacation_'")
        sys.exit(1)
    
    target_directory = sys.argv[1]
    regex_pattern = sys.argv[2]
    replace_with = sys.argv[3]
    
    rename_files(target_directory, regex_pattern, replace_with)