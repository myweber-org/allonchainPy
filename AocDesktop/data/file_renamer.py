
import os
import sys
from datetime import datetime

def rename_files_by_date(directory, prefix="file_"):
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        
        for filename in files:
            filepath = os.path.join(directory, filename)
            mod_time = os.path.getmtime(filepath)
            date_str = datetime.fromtimestamp(mod_time).strftime("%Y%m%d_%H%M%S")
            
            name, ext = os.path.splitext(filename)
            new_filename = f"{prefix}{date_str}{ext}"
            new_filepath = os.path.join(directory, new_filename)
            
            counter = 1
            while os.path.exists(new_filepath):
                new_filename = f"{prefix}{date_str}_{counter}{ext}"
                new_filepath = os.path.join(directory, new_filename)
                counter += 1
            
            os.rename(filepath, new_filepath)
            print(f"Renamed: {filename} -> {new_filename}")
        
        print(f"Successfully renamed {len(files)} files.")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory_path> [prefix]")
        sys.exit(1)
    
    dir_path = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else "file_"
    
    if not os.path.isdir(dir_path):
        print(f"Error: {dir_path} is not a valid directory.")
        sys.exit(1)
    
    rename_files_by_date(dir_path, prefix)
import os
import sys
import argparse

def rename_files(directory, prefix='file', start_number=1, extension=None):
    """
    Rename files in a directory with sequential numbering.
    
    Args:
        directory (str): Path to directory containing files
        prefix (str): Prefix for renamed files
        start_number (int): Starting number for sequence
        extension (str): Filter files by extension (e.g., 'txt', 'jpg')
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return False
    
    try:
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
                except OSError as e:
                    print(f"Error renaming {filename}: {e}")
        
        print(f"\nRenaming complete. {counter - start_number} files renamed.")
        return True
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Rename files with sequential numbering.')
    parser.add_argument('directory', help='Directory containing files to rename')
    parser.add_argument('--prefix', default='file', help='Prefix for renamed files (default: file)')
    parser.add_argument('--start', type=int, default=1, help='Starting number (default: 1)')
    parser.add_argument('--ext', help='Filter by file extension (e.g., jpg, txt)')
    
    args = parser.parse_args()
    
    rename_files(args.directory, args.prefix, args.start, args.ext)

if __name__ == '__main__':
    main()
import os
import sys
from pathlib import Path

def rename_files_sequentially(directory, prefix="file", extension=".txt"):
    try:
        path = Path(directory)
        if not path.exists() or not path.is_dir():
            print(f"Error: Directory '{directory}' does not exist.")
            return
        
        files = [f for f in path.iterdir() if f.is_file()]
        if not files:
            print("No files found in directory.")
            return
        
        sorted_files = sorted(files, key=lambda x: x.stat().st_mtime)
        
        for index, file_path in enumerate(sorted_files, start=1):
            new_name = f"{prefix}_{index:03d}{extension}"
            new_path = file_path.parent / new_name
            
            if new_path.exists():
                print(f"Skipping {file_path.name}: {new_name} already exists.")
                continue
            
            file_path.rename(new_path)
            print(f"Renamed: {file_path.name} -> {new_name}")
    
    except PermissionError:
        print("Permission denied. Try running with appropriate privileges.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory_path> [prefix] [extension]")
        sys.exit(1)
    
    dir_path = sys.argv[1]
    prefix_arg = sys.argv[2] if len(sys.argv) > 2 else "file"
    ext_arg = sys.argv[3] if len(sys.argv) > 3 else ".txt"
    
    rename_files_sequentially(dir_path, prefix_arg, ext_arg)
import os
import sys

def rename_files(directory, prefix="file"):
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        files.sort()
        
        for index, filename in enumerate(files, start=1):
            extension = os.path.splitext(filename)[1]
            new_name = f"{prefix}_{index:03d}{extension}"
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")
            
        print(f"Successfully renamed {len(files)} files.")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory> [prefix]")
        sys.exit(1)
    
    target_dir = sys.argv[1]
    name_prefix = sys.argv[2] if len(sys.argv) > 2 else "file"
    
    if not os.path.isdir(target_dir):
        print(f"Error: {target_dir} is not a valid directory.")
        sys.exit(1)
    
    rename_files(target_dir, name_prefix)