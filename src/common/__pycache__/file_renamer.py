import os
import re
import sys

def rename_files(directory, pattern, replacement):
    """
    Rename files in the specified directory by replacing parts of the filename
    that match the given regex pattern with the replacement string.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return False

    renamed_count = 0
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            new_name = re.sub(pattern, replacement, filename)
            if new_name != filename:
                new_path = os.path.join(directory, new_name)
                try:
                    os.rename(file_path, new_path)
                    print(f"Renamed: '{filename}' -> '{new_name}'")
                    renamed_count += 1
                except OSError as e:
                    print(f"Error renaming '{filename}': {e}")

    print(f"Renaming complete. {renamed_count} files renamed.")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python file_renamer.py <directory> <pattern> <replacement>")
        sys.exit(1)

    target_dir = sys.argv[1]
    regex_pattern = sys.argv[2]
    repl_string = sys.argv[3]

    rename_files(target_dir, regex_pattern, repl_string)