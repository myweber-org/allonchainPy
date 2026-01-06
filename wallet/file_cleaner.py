import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

class TempFileCleaner:
    def __init__(self, target_dir: Optional[str] = None):
        self.target_dir = Path(target_dir) if target_dir else Path(tempfile.gettempdir())
        self.removed_files = []
        self.removed_dirs = []

    def identify_temp_files(self, patterns: List[str] = None) -> List[Path]:
        if patterns is None:
            patterns = ['*.tmp', 'temp_*', '~*', '*.bak']
        
        found_files = []
        for pattern in patterns:
            found_files.extend(self.target_dir.glob(pattern))
        return found_files

    def cleanup_files(self, patterns: List[str] = None, dry_run: bool = False) -> dict:
        files_to_remove = self.identify_temp_files(patterns)
        stats = {'files_found': len(files_to_remove), 'files_removed': 0, 'size_freed': 0}
        
        for file_path in files_to_remove:
            if dry_run:
                print(f"[DRY RUN] Would remove: {file_path}")
                continue
                
            try:
                file_size = file_path.stat().st_size
                file_path.unlink()
                self.removed_files.append(file_path)
                stats['files_removed'] += 1
                stats['size_freed'] += file_size
                print(f"Removed: {file_path} ({file_size} bytes)")
            except Exception as e:
                print(f"Error removing {file_path}: {e}")
        
        return stats

    def cleanup_empty_dirs(self, dry_run: bool = False) -> int:
        empty_dirs_removed = 0
        for root, dirs, files in os.walk(self.target_dir, topdown=False):
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                try:
                    if not any(dir_path.iterdir()):
                        if dry_run:
                            print(f"[DRY RUN] Would remove empty directory: {dir_path}")
                        else:
                            dir_path.rmdir()
                            self.removed_dirs.append(dir_path)
                            empty_dirs_removed += 1
                            print(f"Removed empty directory: {dir_path}")
                except Exception as e:
                    print(f"Error processing directory {dir_path}: {e}")
        
        return empty_dirs_removed

    def get_summary(self) -> dict:
        total_size = sum(f.stat().st_size for f in self.removed_files if f.exists())
        return {
            'target_directory': str(self.target_dir),
            'files_removed': len(self.removed_files),
            'directories_removed': len(self.removed_dirs),
            'total_space_freed': total_size
        }

def main():
    cleaner = TempFileCleaner()
    print(f"Cleaning temporary files in: {cleaner.target_dir}")
    
    file_stats = cleaner.cleanup_files()
    dir_count = cleaner.cleanup_empty_dirs()
    
    summary = cleaner.get_summary()
    print("\nCleanup Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()