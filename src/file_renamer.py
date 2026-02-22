
import os
import sys

def rename_files(directory, prefix):
    try:
        if not os.path.isdir(directory):
            print(f"Error: {directory} is not a valid directory.")
            return False

        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

        if not files:
            print("No files found in the directory.")
            return True

        for filename in files:
            new_name = f"{prefix}_{filename}"
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)

            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_name}")
            except OSError as e:
                print(f"Error renaming {filename}: {e}")

        return True

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python file_renamer.py <directory_path> <prefix>")
        sys.exit(1)

    dir_path = sys.argv[1]
    prefix = sys.argv[2]

    success = rename_files(dir_path, prefix)
    sys.exit(0 if success else 1)
import os
import sys

def rename_files_with_sequence(directory, prefix="file", extension=".txt"):
    """
    Rename all files in the specified directory with sequential numbering.
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
            return False

    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory> [prefix] [extension]")
        sys.exit(1)

    target_dir = sys.argv[1]
    prefix_arg = sys.argv[2] if len(sys.argv) > 2 else "file"
    extension_arg = sys.argv[3] if len(sys.argv) > 3 else ".txt"

    success = rename_files_with_sequence(target_dir, prefix_arg, extension_arg)
    if success:
        print("File renaming completed successfully.")
    else:
        print("File renaming encountered errors.")