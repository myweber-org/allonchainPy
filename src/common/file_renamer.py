
import os
import glob
from pathlib import Path
from datetime import datetime

def rename_files_with_timestamp(directory, prefix="file", extension=".txt"):
    files = glob.glob(os.path.join(directory, f"*{extension}"))
    files.sort(key=os.path.getctime)
    
    for index, filepath in enumerate(files, start=1):
        creation_time = datetime.fromtimestamp(os.path.getctime(filepath))
        timestamp = creation_time.strftime("%Y%m%d_%H%M%S")
        new_filename = f"{prefix}_{timestamp}_{index:03d}{extension}"
        new_filepath = os.path.join(directory, new_filename)
        
        try:
            os.rename(filepath, new_filepath)
            print(f"Renamed: {Path(filepath).name} -> {new_filename}")
        except OSError as e:
            print(f"Error renaming {filepath}: {e}")

if __name__ == "__main__":
    target_directory = "./documents"
    if os.path.exists(target_directory):
        rename_files_with_timestamp(target_directory, prefix="document", extension=".pdf")
    else:
        print(f"Directory '{target_directory}' does not exist.")