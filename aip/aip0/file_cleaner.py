
import os
import shutil
import tempfile
from pathlib import Path

def clean_temp_files(directory: str, extensions: tuple = ('.tmp', '.temp', '.log'), max_age_days: int = 7):
    """
    Remove temporary files with specified extensions older than a given number of days.
    
    Args:
        directory: Path to the directory to clean.
        extensions: Tuple of file extensions to consider as temporary.
        max_age_days: Maximum age of files in days before removal.
    """
    target_dir = Path(directory)
    if not target_dir.exists() or not target_dir.is_dir():
        raise ValueError(f"Invalid directory: {directory}")
    
    current_time = os.path.getctime if hasattr(os.path, 'getctime') else os.path.getmtime
    cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
    
    removed_count = 0
    total_size = 0
    
    for item in target_dir.rglob('*'):
        if item.is_file() and item.suffix.lower() in extensions:
            try:
                file_time = current_time(item)
                if file_time < cutoff_time:
                    file_size = item.stat().st_size
                    item.unlink()
                    removed_count += 1
                    total_size += file_size
                    print(f"Removed: {item.name} ({file_size} bytes)")
            except (OSError, PermissionError) as e:
                print(f"Failed to remove {item}: {e}")
    
    print(f"Cleaning complete. Removed {removed_count} files, freed {total_size} bytes.")

def create_sample_temp_files(test_dir: str, count: int = 5):
    """Create sample temporary files for testing."""
    test_path = Path(test_dir)
    test_path.mkdir(exist_ok=True)
    
    for i in range(count):
        temp_file = test_path / f"temp_file_{i}.tmp"
        temp_file.write_text(f"Temporary content {i}")
        # Set old modification time
        old_time = time.time() - (10 * 24 * 60 * 60)
        os.utime(temp_file, (old_time, old_time))
    
    print(f"Created {count} sample temp files in {test_dir}")

if __name__ == "__main__":
    import time
    
    # Example usage
    sample_dir = tempfile.mkdtemp(prefix="cleaner_test_")
    print(f"Test directory: {sample_dir}")
    
    try:
        create_sample_temp_files(sample_dir, 3)
        clean_temp_files(sample_dir, max_age_days=5)
    finally:
        # Cleanup test directory
        if Path(sample_dir).exists():
            shutil.rmtree(sample_dir)
            print(f"Cleaned up test directory: {sample_dir}")