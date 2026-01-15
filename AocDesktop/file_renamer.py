
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
        extension (str): Filter files by extension (optional)
    """
    try:
        if not os.path.isdir(directory):
            print(f"Error: Directory '{directory}' does not exist.")
            return False
        
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
        
        print(f"\nSuccessfully renamed {counter - start_number} files.")
        return True
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Rename files with sequential numbering.')
    parser.add_argument('directory', help='Directory containing files to rename')
    parser.add_argument('prefix', help='Prefix for renamed files')
    parser.add_argument('--start', type=int, default=1, help='Starting number (default: 1)')
    parser.add_argument('--ext', help='Filter by file extension (e.g., jpg, png)')
    
    args = parser.parse_args()
    
    success = rename_files(args.directory, args.prefix, args.start, args.ext)
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()
import os
import sys

def rename_files_with_sequence(directory, prefix="file"):
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
        return True
        
    except Exception as e:
        print(f"Error occurred: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory_path> [prefix]")
        sys.exit(1)
    
    dir_path = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else "file"
    
    if not os.path.isdir(dir_path):
        print(f"Error: {dir_path} is not a valid directory.")
        sys.exit(1)
    
    rename_files_with_sequence(dir_path, prefix)