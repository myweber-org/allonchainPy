
import os
import shutil
import tempfile
from pathlib import Path

def clean_temporary_files(directory_path, extensions=None, days_old=7):
    """
    Remove temporary files from a specified directory.
    Files can be filtered by extension and age.
    """
    if extensions is None:
        extensions = ['.tmp', '.temp', '.log', '.cache']
    
    target_dir = Path(directory_path)
    if not target_dir.exists() or not target_dir.is_dir():
        raise ValueError(f"Invalid directory: {directory_path}")
    
    current_time = os.path.getctime(target_dir)
    removed_count = 0
    
    for item in target_dir.iterdir():
        if item.is_file():
            file_age = current_time - os.path.getctime(item)
            if file_age > days_old * 86400:
                if any(item.suffix.lower() in ext for ext in extensions):
                    try:
                        item.unlink()
                        removed_count += 1
                        print(f"Removed: {item.name}")
                    except OSError as e:
                        print(f"Error removing {item.name}: {e}")
    
    return removed_count

def create_sample_temporary_files():
    """Helper function to create sample temporary files for testing."""
    temp_dir = tempfile.gettempdir()
    sample_files = [
        "test_file.tmp",
        "another_file.temp",
        "log_file.log",
        "cache_file.cache",
        "keep_file.txt"
    ]
    
    for filename in sample_files:
        filepath = os.path.join(temp_dir, filename)
        with open(filepath, 'w') as f:
            f.write("Temporary content")
    
    return temp_dir

if __name__ == "__main__":
    try:
        sample_dir = create_sample_temporary_files()
        print(f"Created sample files in: {sample_dir}")
        
        cleaned = clean_temporary_files(sample_dir)
        print(f"Cleaned {cleaned} temporary files")
        
    except Exception as e:
        print(f"An error occurred: {e}")