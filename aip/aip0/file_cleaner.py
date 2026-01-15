
import os
import shutil
import tempfile
from pathlib import Path

def clean_temporary_files(directory: str, extensions: tuple = ('.tmp', '.temp', '.log'), dry_run: bool = False) -> dict:
    """
    Remove temporary files with specified extensions from a directory.
    Returns a dictionary with statistics.
    """
    if not os.path.isdir(directory):
        raise ValueError(f"Directory does not exist: {directory}")

    stats = {
        'total_found': 0,
        'total_removed': 0,
        'failed_removals': [],
        'removed_files': []
    }

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extensions):
                file_path = os.path.join(root, file)
                stats['total_found'] += 1

                if not dry_run:
                    try:
                        os.remove(file_path)
                        stats['total_removed'] += 1
                        stats['removed_files'].append(file_path)
                    except Exception as e:
                        stats['failed_removals'].append((file_path, str(e)))
                else:
                    stats['removed_files'].append(file_path)

    return stats

def create_sample_temporary_files(test_dir: str, count: int = 5) -> None:
    """Create sample temporary files for testing."""
    Path(test_dir).mkdir(parents=True, exist_ok=True)
    
    for i in range(count):
        temp_name = f"temp_file_{i}.tmp"
        temp_path = os.path.join(test_dir, temp_name)
        with open(temp_path, 'w') as f:
            f.write(f"Temporary content {i}")

def main():
    """Example usage of the file cleaner."""
    test_directory = tempfile.mkdtemp(prefix="cleaner_test_")
    print(f"Created test directory: {test_directory}")

    create_sample_temporary_files(test_directory, 3)
    create_sample_temporary_files(os.path.join(test_directory, "subdir"), 2)

    print("\n--- Dry run ---")
    result = clean_temporary_files(test_directory, dry_run=True)
    print(f"Files found: {result['total_found']}")
    print(f"Files to be removed: {result['removed_files']}")

    print("\n--- Actual cleanup ---")
    result = clean_temporary_files(test_directory, dry_run=False)
    print(f"Files removed: {result['total_removed']}")
    print(f"Failed removals: {len(result['failed_removals'])}")

    shutil.rmtree(test_directory)
    print(f"\nCleaned up test directory: {test_directory}")

if __name__ == "__main__":
    main()