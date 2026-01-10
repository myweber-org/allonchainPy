
import os
import shutil
import tempfile
from pathlib import Path

def clean_temporary_files(directory_path, extensions=None, dry_run=False):
    """
    Remove temporary files from a specified directory.
    
    Args:
        directory_path (str or Path): Path to the directory to clean.
        extensions (list, optional): List of temporary file extensions to target.
            If None, uses common temporary extensions.
        dry_run (bool): If True, only print files to be removed without deleting.
    
    Returns:
        int: Number of files removed (or would be removed in dry run).
    """
    if extensions is None:
        extensions = ['.tmp', '.temp', '.bak', '.swp', '~']
    
    target_dir = Path(directory_path)
    if not target_dir.exists() or not target_dir.is_dir():
        raise ValueError(f"Invalid directory: {directory_path}")
    
    removed_count = 0
    
    for item in target_dir.rglob('*'):
        if not item.is_file():
            continue
        
        if any(item.name.endswith(ext) for ext in extensions):
            if dry_run:
                print(f"[DRY RUN] Would remove: {item}")
            else:
                try:
                    item.unlink()
                    print(f"Removed: {item}")
                except OSError as e:
                    print(f"Error removing {item}: {e}")
                    continue
            removed_count += 1
    
    return removed_count

def create_test_environment():
    """Create a test directory with temporary files for demonstration."""
    test_dir = tempfile.mkdtemp(prefix='clean_test_')
    print(f"Created test directory: {test_dir}")
    
    test_files = [
        'document.txt',
        'backup.bak',
        'temp.tmp',
        'swapfile.swp',
        'cache.temp',
        'config~',
        'data.csv',
        'log.tmp.log'
    ]
    
    for filename in test_files:
        filepath = Path(test_dir) / filename
        filepath.touch()
    
    return test_dir

if __name__ == "__main__":
    # Example usage
    test_dir = create_test_environment()
    
    print("\n--- Dry run ---")
    count = clean_temporary_files(test_dir, dry_run=True)
    print(f"Would remove {count} files")
    
    print("\n--- Actual cleanup ---")
    count = clean_temporary_files(test_dir)
    print(f"Removed {count} files")
    
    # Cleanup test directory
    shutil.rmtree(test_dir)
    print(f"\nCleaned up test directory: {test_dir}")