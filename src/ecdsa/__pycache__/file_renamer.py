
import os
import sys

def rename_files_with_sequence(directory, prefix="file", start_number=1, extension=".txt"):
    """
    Rename all files in the specified directory with sequential numbering.
    
    Args:
        directory (str): Path to the directory containing files to rename.
        prefix (str): Prefix for the new filenames.
        start_number (int): Starting number for the sequence.
        extension (str): File extension to filter and apply.
    
    Returns:
        int: Number of files successfully renamed.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return 0
    
    try:
        files = [f for f in os.listdir(directory) 
                 if os.path.isfile(os.path.join(directory, f)) and f.endswith(extension)]
        files.sort()
        
        renamed_count = 0
        for index, filename in enumerate(files, start=start_number):
            old_path = os.path.join(directory, filename)
            new_filename = f"{prefix}_{index:03d}{extension}"
            new_path = os.path.join(directory, new_filename)
            
            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")
                renamed_count += 1
            except OSError as e:
                print(f"Failed to rename {filename}: {e}")
        
        print(f"\nSuccessfully renamed {renamed_count} files.")
        return renamed_count
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory> [prefix] [start_number] [extension]")
        print("Example: python file_renamer.py ./documents document 1 .txt")
        sys.exit(1)
    
    dir_path = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else "file"
    start_num = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    ext = sys.argv[4] if len(sys.argv) > 4 else ".txt"
    
    rename_files_with_sequence(dir_path, prefix, start_num, ext)