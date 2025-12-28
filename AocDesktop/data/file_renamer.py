
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