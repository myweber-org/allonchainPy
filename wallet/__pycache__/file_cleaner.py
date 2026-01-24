import os
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_old_files(directory_path, days_threshold=7):
    """
    Remove files older than specified days from given directory.
    """
    if not os.path.exists(directory_path):
        logger.error(f"Directory does not exist: {directory_path}")
        return
    
    current_time = time.time()
    threshold_seconds = days_threshold * 24 * 60 * 60
    
    removed_count = 0
    for item in Path(directory_path).iterdir():
        if item.is_file():
            file_age = current_time - item.stat().st_mtime
            if file_age > threshold_seconds:
                try:
                    item.unlink()
                    logger.info(f"Removed: {item}")
                    removed_count += 1
                except OSError as e:
                    logger.error(f"Failed to remove {item}: {e}")
    
    logger.info(f"Cleanup completed. Removed {removed_count} file(s).")

if __name__ == "__main__":
    target_dir = "/tmp/test_files"
    clean_old_files(target_dir)