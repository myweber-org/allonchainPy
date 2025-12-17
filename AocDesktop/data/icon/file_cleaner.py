
import os
import shutil
import tempfile
from pathlib import Path

def clean_temporary_files(directory_path, extensions=None, days_old=7):
    """
    Remove temporary files from a specified directory.
    
    Args:
        directory_path (str or Path): Path to the directory to clean.
        extensions (list, optional): List of file extensions to target.
            If None, uses common temporary extensions.
        days_old (int, optional): Only remove files older than this many days.
    """
    if extensions is None:
        extensions = ['.tmp', '.temp', '.log', '.cache', '.bak']
    
    target_dir = Path(directory_path)
    if not target_dir.exists() or not target_dir.is_dir():
        raise ValueError(f"Invalid directory: {directory_path}")
    
    current_time = time.time()
    cutoff_time = current_time - (days_old * 24 * 60 * 60)
    
    removed_count = 0
    total_size = 0
    
    for file_path in target_dir.rglob('*'):
        if file_path.is_file():
            file_ext = file_path.suffix.lower()
            
            if file_ext in extensions or any(
                file_path.name.startswith(prefix) 
                for prefix in ['~$', '.~']
            ):
                try:
                    file_stat = file_path.stat()
                    
                    if file_stat.st_mtime < cutoff_time:
                        file_size = file_stat.st_size
                        file_path.unlink()
                        
                        removed_count += 1
                        total_size += file_size
                        
                        print(f"Removed: {file_path.name} ({file_size} bytes)")
                        
                except (PermissionError, OSError) as e:
                    print(f"Failed to remove {file_path.name}: {e}")
    
    print(f"\nCleaning complete:")
    print(f"  Files removed: {removed_count}")
    print(f"  Total space freed: {total_size / (1024*1024):.2f} MB")

def create_test_environment():
    """Create a test directory with temporary files for demonstration."""
    test_dir = Path(tempfile.mkdtemp(prefix="clean_test_"))
    
    test_files = [
        "document.tmp",
        "backup.bak",
        "~$recent.docx",
        "cache.data",
        "system.log",
        "important.txt",
        "image.png"
    ]
    
    for filename in test_files:
        file_path = test_dir / filename
        file_path.write_text("Sample content for testing.")
        
        old_time = time.time() - (10 * 24 * 60 * 60)
        os.utime(file_path, (old_time, old_time))
    
    return test_dir

if __name__ == "__main__":
    import time
    
    print("Testing file cleaner utility...")
    
    test_directory = create_test_environment()
    print(f"Created test directory: {test_directory}")
    
    try:
        clean_temporary_files(
            directory_path=test_directory,
            extensions=['.tmp', '.bak', '.log'],
            days_old=5
        )
    finally:
        if test_directory.exists():
            shutil.rmtree(test_directory)
            print(f"\nCleaned up test directory: {test_directory}")