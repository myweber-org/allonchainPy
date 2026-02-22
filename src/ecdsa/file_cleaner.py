import os
import shutil
import argparse

def clean_directory(directory, extensions_to_remove, dry_run=False):
    """
    Remove files with specified extensions from the given directory.
    If dry_run is True, only print the files that would be removed.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    removed_count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions_to_remove):
                file_path = os.path.join(root, file)
                if dry_run:
                    print(f"[DRY RUN] Would remove: {file_path}")
                else:
                    try:
                        os.remove(file_path)
                        print(f"Removed: {file_path}")
                        removed_count += 1
                    except OSError as e:
                        print(f"Error removing {file_path}: {e}")

    if dry_run:
        print(f"[DRY RUN] Total files that would be removed: {removed_count}")
    else:
        print(f"Total files removed: {removed_count}")

def main():
    parser = argparse.ArgumentParser(description="Clean temporary files from a directory.")
    parser.add_argument("directory", help="Directory to clean")
    parser.add_argument("-e", "--extensions", nargs="+", default=[".tmp", ".log", ".bak"],
                        help="File extensions to remove (default: .tmp .log .bak)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Perform a dry run without deleting files")

    args = parser.parse_args()

    clean_directory(args.directory, args.extensions, args.dry_run)

if __name__ == "__main__":
    main()