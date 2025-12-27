import os
import time
import shutil

def clean_old_files(directory, days=7):
    """
    Remove files in the specified directory that are older than the given number of days.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    current_time = time.time()
    cutoff_time = current_time - (days * 24 * 60 * 60)
    removed_count = 0
    error_count = 0

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        try:
            if os.path.isfile(filepath):
                file_mtime = os.path.getmtime(filepath)
                if file_mtime < cutoff_time:
                    os.remove(filepath)
                    removed_count += 1
                    print(f"Removed: {filepath}")
            elif os.path.isdir(filepath):
                dir_mtime = os.path.getmtime(filepath)
                if dir_mtime < cutoff_time:
                    shutil.rmtree(filepath)
                    removed_count += 1
                    print(f"Removed directory: {filepath}")
        except Exception as e:
            error_count += 1
            print(f"Error processing {filepath}: {e}")

    print(f"Cleanup completed. Removed {removed_count} items. Encountered {error_count} errors.")

if __name__ == "__main__":
    target_dir = "/tmp/my_app_cache"
    clean_old_files(target_dir, days=7)