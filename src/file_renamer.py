
import os
import sys

def rename_files_with_sequence(directory, prefix="file", extension=".txt"):
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        files.sort()
        
        for index, filename in enumerate(files, start=1):
            old_path = os.path.join(directory, filename)
            new_name = f"{prefix}_{index:03d}{extension}"
            new_path = os.path.join(directory, new_name)
            
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")
            
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory> [prefix] [extension]")
        sys.exit(1)
    
    target_dir = sys.argv[1]
    prefix_arg = sys.argv[2] if len(sys.argv) > 2 else "file"
    extension_arg = sys.argv[3] if len(sys.argv) > 3 else ".txt"
    
    if not os.path.isdir(target_dir):
        print(f"Error: {target_dir} is not a valid directory")
        sys.exit(1)
    
    success = rename_files_with_sequence(target_dir, prefix_arg, extension_arg)
    if success:
        print("File renaming completed successfully")
    else:
        print("File renaming failed")