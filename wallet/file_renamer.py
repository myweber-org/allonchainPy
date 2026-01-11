
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