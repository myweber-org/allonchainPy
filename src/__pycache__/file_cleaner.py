
import os
import shutil
import tempfile
from pathlib import Path

class TempFileCleaner:
    def __init__(self, target_dir=None, patterns=None):
        self.target_dir = Path(target_dir) if target_dir else Path(tempfile.gettempdir())
        self.patterns = patterns or ['.tmp', '.temp', '_temp', 'temp_', '~']
        self.removed_files = []
        self.removed_dirs = []

    def is_temp_file(self, filename):
        name_lower = filename.lower()
        return any(pattern in name_lower for pattern in self.patterns)

    def scan_and_clean(self, dry_run=False):
        for item in self.target_dir.rglob('*'):
            if item.is_file() and self.is_temp_file(item.name):
                if not dry_run:
                    try:
                        item.unlink()
                        self.removed_files.append(str(item))
                    except (PermissionError, OSError) as e:
                        print(f"Failed to remove {item}: {e}")
                else:
                    print(f"[Dry Run] Would remove file: {item}")

            elif item.is_dir() and self.is_temp_file(item.name):
                if not dry_run:
                    try:
                        shutil.rmtree(item)
                        self.removed_dirs.append(str(item))
                    except (PermissionError, OSError) as e:
                        print(f"Failed to remove directory {item}: {e}")
                else:
                    print(f"[Dry Run] Would remove directory: {item}")

        return {
            'files_removed': len(self.removed_files),
            'dirs_removed': len(self.removed_dirs),
            'total_freed': self._estimate_freed_space()
        }

    def _estimate_freed_space(self):
        total_size = 0
        for file_path in self.removed_files:
            try:
                total_size += Path(file_path).stat().st_size
            except OSError:
                continue
        return total_size

    def get_report(self):
        return {
            'target_directory': str(self.target_dir),
            'patterns_used': self.patterns,
            'files_removed': self.removed_files,
            'directories_removed': self.removed_dirs
        }

def main():
    cleaner = TempFileCleaner()
    print(f"Cleaning temporary files in: {cleaner.target_dir}")
    
    result = cleaner.scan_and_clean(dry_run=True)
    print(f"Dry run completed. Would remove {result['files_removed']} files and {result['dirs_removed']} directories.")
    
    response = input("Proceed with actual cleanup? (yes/no): ")
    if response.lower() == 'yes':
        result = cleaner.scan_and_clean(dry_run=False)
        print(f"Cleanup completed. Removed {result['files_removed']} files and {result['dirs_removed']} directories.")
        print(f"Estimated space freed: {result['total_freed'] / (1024*1024):.2f} MB")
        
        report = cleaner.get_report()
        if report['files_removed'] or report['directories_removed']:
            print("\nDetailed report saved to cleanup_report.txt")
            with open('cleanup_report.txt', 'w') as f:
                f.write(f"Temp File Cleanup Report\n")
                f.write(f"Target Directory: {report['target_directory']}\n")
                f.write(f"Patterns Used: {', '.join(report['patterns_used'])}\n\n")
                f.write("Removed Files:\n")
                for file in report['files_removed']:
                    f.write(f"  - {file}\n")
                f.write("\nRemoved Directories:\n")
                    for dir in report['directories_removed']:
                        f.write(f"  - {dir}\n")
    else:
        print("Cleanup cancelled.")

if __name__ == "__main__":
    main()