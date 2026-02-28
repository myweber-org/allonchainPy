
import os
import re
import sys

def rename_files(directory, pattern, replacement):
    """
    Rename files in the specified directory based on a regex pattern.
    
    Args:
        directory (str): Path to the directory containing files to rename.
        pattern (str): Regex pattern to match in filenames.
        replacement (str): Replacement string for matched pattern.
    """
    try:
        if not os.path.isdir(directory):
            print(f"Error: Directory '{directory}' does not exist.")
            return False

        files = os.listdir(directory)
        renamed_count = 0

        for filename in files:
            file_path = os.path.join(directory, filename)
            
            if os.path.isfile(file_path):
                new_name = re.sub(pattern, replacement, filename)
                
                if new_name != filename:
                    new_path = os.path.join(directory, new_name)
                    
                    if os.path.exists(new_path):
                        print(f"Warning: Cannot rename '{filename}' to '{new_name}' - target already exists.")
                        continue
                    
                    try:
                        os.rename(file_path, new_path)
                        print(f"Renamed: '{filename}' -> '{new_name}'")
                        renamed_count += 1
                    except OSError as e:
                        print(f"Error renaming '{filename}': {e}")
        
        print(f"\nRenaming complete. {renamed_count} files renamed.")
        return True
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

def main():
    if len(sys.argv) != 4:
        print("Usage: python file_renamer.py <directory> <pattern> <replacement>")
        print("Example: python file_renamer.py ./photos 'IMG_\\d+' 'Vacation_'")
        sys.exit(1)
    
    directory = sys.argv[1]
    pattern = sys.argv[2]
    replacement = sys.argv[3]
    
    rename_files(directory, pattern, replacement)

if __name__ == "__main__":
    main()import os
import sys

def rename_files_with_sequence(directory, prefix="file", extension=".txt"):
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        files.sort()
        
        for index, filename in enumerate(files, start=1):
            old_path = os.path.join(directory, filename)
            new_filename = f"{prefix}_{index:03d}{extension}"
            new_path = os.path.join(directory, new_filename)
            
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")
        
        print(f"Successfully renamed {len(files)} files.")
        return True
        
    except Exception as e:
        print(f"Error occurred: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
        prefix = sys.argv[2] if len(sys.argv) > 2 else "file"
        extension = sys.argv[3] if len(sys.argv) > 3 else ".txt"
        
        rename_files_with_sequence(target_dir, prefix, extension)
    else:
        print("Usage: python file_renamer.py <directory> [prefix] [extension]")
        print("Example: python file_renamer.py ./documents image .jpg")
import os
import sys

def batch_rename_files(directory, prefix, start_number=1):
    """
    Rename all files in a directory with sequential numbering.
    
    Args:
        directory (str): Path to the directory containing files to rename
        prefix (str): Prefix for the new filenames
        start_number (int): Starting number for the sequence
    """
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return False
    
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a directory.")
        return False
    
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        files.sort()
        
        if not files:
            print("No files found in the directory.")
            return True
        
        print(f"Found {len(files)} files to rename.")
        print("Preview of changes:")
        
        for i, filename in enumerate(files, start=start_number):
            file_ext = os.path.splitext(filename)[1]
            new_name = f"{prefix}_{i:03d}{file_ext}"
            print(f"  {filename} -> {new_name}")
        
        confirmation = input("\nProceed with renaming? (y/n): ").strip().lower()
        
        if confirmation != 'y':
            print("Operation cancelled.")
            return False
        
        for i, filename in enumerate(files, start=start_number):
            file_ext = os.path.splitext(filename)[1]
            new_name = f"{prefix}_{i:03d}{file_ext}"
            
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")
        
        print(f"\nSuccessfully renamed {len(files)} files.")
        return True
        
    except Exception as e:
        print(f"Error during renaming: {e}")
        return False

def main():
    if len(sys.argv) < 3:
        print("Usage: python file_renamer.py <directory> <prefix> [start_number]")
        print("Example: python file_renamer.py ./photos vacation_ 1")
        return
    
    directory = sys.argv[1]
    prefix = sys.argv[2]
    start_number = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    
    batch_rename_files(directory, prefix, start_number)

if __name__ == "__main__":
    main()