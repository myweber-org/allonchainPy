import os
import sys

def batch_rename(directory, prefix, extension):
    """
    Rename all files in a directory with sequential numbering.
    
    Args:
        directory (str): Path to the directory containing files
        prefix (str): Prefix for the new filenames
        extension (str): File extension (without dot)
    """
    try:
        if not os.path.exists(directory):
            print(f"Error: Directory '{directory}' does not exist.")
            return False
        
        files = [f for f in os.listdir(directory) 
                if os.path.isfile(os.path.join(directory, f))]
        
        if not files:
            print("No files found in the directory.")
            return False
        
        files.sort()
        
        for index, filename in enumerate(files, start=1):
            old_path = os.path.join(directory, filename)
            new_name = f"{prefix}_{index:03d}.{extension}"
            new_path = os.path.join(directory, new_name)
            
            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_name}")
            except Exception as e:
                print(f"Failed to rename {filename}: {str(e)}")
        
        print(f"Renaming complete. Processed {len(files)} files.")
        return True
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False

def main():
    if len(sys.argv) != 4:
        print("Usage: python file_renamer.py <directory> <prefix> <extension>")
        print("Example: python file_renamer.py ./photos vacation jpg")
        sys.exit(1)
    
    directory = sys.argv[1]
    prefix = sys.argv[2]
    extension = sys.argv[3]
    
    batch_rename(directory, prefix, extension)

if __name__ == "__main__":
    main()