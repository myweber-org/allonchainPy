
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
import sys
import os

def remove_duplicate_lines(input_file, output_file=None):
    """
    Remove duplicate lines from a text file while preserving order.
    
    Args:
        input_file: Path to the input file
        output_file: Path to the output file (optional, defaults to input_file with '_cleaned' suffix)
    
    Returns:
        Number of duplicate lines removed
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found")
    
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_cleaned{ext}"
    
    seen_lines = set()
    unique_lines = []
    duplicate_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line not in seen_lines:
                seen_lines.add(line)
                unique_lines.append(line)
            else:
                duplicate_count += 1
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(unique_lines)
    
    return duplicate_count

def main():
    if len(sys.argv) < 2:
        print("Usage: python file_cleaner.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        removed = remove_duplicate_lines(input_file, output_file)
        out_path = output_file if output_file else f"{os.path.splitext(input_file)[0]}_cleaned{os.path.splitext(input_file)[1]}"
        print(f"Removed {removed} duplicate lines. Cleaned file saved to: {out_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()