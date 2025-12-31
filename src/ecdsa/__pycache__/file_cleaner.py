
import os
import shutil
import argparse

def clean_directory(directory, extensions=None, dry_run=False):
    """
    Remove files with given extensions from the specified directory.
    If extensions is None, remove all files.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    removed_count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if extensions is None or any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                if dry_run:
                    print(f"[DRY RUN] Would remove: {file_path}")
                else:
                    try:
                        os.remove(file_path)
                        print(f"Removed: {file_path}")
                    except Exception as e:
                        print(f"Failed to remove {file_path}: {e}")
                removed_count += 1

    print(f"Total files removed: {removed_count}")

def main():
    parser = argparse.ArgumentParser(description="Clean temporary files from a directory.")
    parser.add_argument("directory", help="Directory to clean")
    parser.add_argument("-e", "--extensions", nargs="+", help="File extensions to remove (e.g., .tmp .log)")
    parser.add_argument("-d", "--dry-run", action="store_true", help="Simulate removal without deleting files")

    args = parser.parse_args()

    clean_directory(args.directory, args.extensions, args.dry_run)

if __name__ == "__main__":
    main()