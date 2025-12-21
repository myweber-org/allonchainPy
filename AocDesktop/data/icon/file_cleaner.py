
import os
import shutil
import tempfile

def clean_temp_files(directory, days_old=7):
    """
    Remove temporary files older than a specified number of days.
    
    Args:
        directory (str): Path to the directory to clean.
        days_old (int): Remove files older than this many days.
    """
    import time
    current_time = time.time()
    cutoff_time = current_time - (days_old * 24 * 60 * 60)
    
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                file_stat = os.stat(file_path)
                if file_stat.st_mtime < cutoff_time:
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
            except (OSError, PermissionError) as e:
                print(f"Error removing {file_path}: {e}")

def create_sample_temp_files():
    """Create sample temporary files for testing."""
    temp_dir = tempfile.mkdtemp()
    print(f"Created temp directory: {temp_dir}")
    
    for i in range(5):
        temp_file = tempfile.NamedTemporaryFile(
            dir=temp_dir,
            delete=False,
            suffix=f"_sample_{i}.tmp"
        )
        temp_file.write(b"Sample temporary data")
        temp_file.close()
    
    return temp_dir

if __name__ == "__main__":
    sample_dir = create_sample_temp_files()
    print(f"\nCleaning files older than 1 day in {sample_dir}")
    clean_temp_files(sample_dir, days_old=1)
    
    try:
        shutil.rmtree(sample_dir)
        print(f"\nCleaned up test directory: {sample_dir}")
    except Exception as e:
        print(f"Error cleaning test directory: {e}")