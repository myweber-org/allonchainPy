
import os
import sys
from pathlib import Path
from datetime import datetime

def rename_files_sequentially(directory_path, prefix="file"):
    try:
        path = Path(directory_path)
        if not path.exists() or not path.is_dir():
            print(f"Error: {directory_path} is not a valid directory")
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
        
        counter = 1
        for _, file_path in files:
            extension = file_path.suffix
            new_name = f"{prefix}_{counter:03d}{extension}"
            new_path = file_path.parent / new_name
            
            try:
                file_path.rename(new_path)
                print(f"Renamed: {file_path.name} -> {new_name}")
                counter += 1
            except OSError as e:
                print(f"Failed to rename {file_path.name}: {e}")
        
        print(f"Successfully renamed {counter-1} files")
        return True
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory_path> [prefix]")
        sys.exit(1)
    
    dir_path = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else "file"
    
    rename_files_sequentially(dir_path, prefix)
import os
import sys
import argparse

def rename_files(directory, prefix, start_number=1, extension=None):
    """
    Rename files in a directory with sequential numbering.
    
    Args:
        directory (str): Path to directory containing files
        prefix (str): Prefix for renamed files
        start_number (int): Starting number for sequence
        extension (str): Filter by file extension (optional)
    """
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return
    
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a directory.")
        return
    
    files = os.listdir(directory)
    
    if extension:
        files = [f for f in files if f.lower().endswith(f'.{extension.lower()}')]
    
    files.sort()
    
    counter = start_number
    
    for filename in files:
        old_path = os.path.join(directory, filename)
        
        if os.path.isfile(old_path):
            file_ext = os.path.splitext(filename)[1]
            new_filename = f"{prefix}_{counter:03d}{file_ext}"
            new_path = os.path.join(directory, new_filename)
            
            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")
                counter += 1
            except Exception as e:
                print(f"Error renaming {filename}: {e}")
    
    print(f"\nRenaming complete. {counter - start_number} files renamed.")

def main():
    parser = argparse.ArgumentParser(description='Rename files with sequential numbering.')
    parser.add_argument('directory', help='Directory containing files to rename')
    parser.add_argument('prefix', help='Prefix for renamed files')
    parser.add_argument('--start', type=int, default=1, help='Starting number (default: 1)')
    parser.add_argument('--ext', help='Filter by file extension (e.g., jpg, png)')
    
    args = parser.parse_args()
    
    rename_files(args.directory, args.prefix, args.start, args.ext)

if __name__ == '__main__':
    main()