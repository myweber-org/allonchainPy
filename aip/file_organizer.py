
import os
import shutil
from pathlib import Path

def organize_files(directory="."):
    """
    Organizes files in the specified directory by moving them into
    subdirectories based on their file extensions.
    """
    base_path = Path(directory).resolve()
    
    # Define categories and their associated file extensions
    categories = {
        "Images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp"],
        "Documents": [".pdf", ".docx", ".txt", ".xlsx", ".pptx", ".md", ".rtf"],
        "Archives": [".zip", ".tar", ".gz", ".rar", ".7z"],
        "Code": [".py", ".js", ".html", ".css", ".java", ".cpp", ".c", ".json"],
        "Audio": [".mp3", ".wav", ".flac", ".aac", ".ogg"],
        "Video": [".mp4", ".avi", ".mkv", ".mov", ".wmv"],
        "Executables": [".exe", ".msi", ".sh", ".bat", ".app"],
    }
    
    # Create category folders if they don't exist
    for category in categories:
        category_path = base_path / category
        category_path.mkdir(exist_ok=True)
    
    # Track moved files
    moved_files = []
    skipped_files = []
    
    # Iterate over items in the directory
    for item in base_path.iterdir():
        if item.is_file():
            file_ext = item.suffix.lower()
            moved = False
            
            # Find the appropriate category for the file
            for category, extensions in categories.items():
                if file_ext in extensions:
                    target_path = base_path / category / item.name
                    
                    # Handle naming conflicts
                    counter = 1
                    while target_path.exists():
                        stem = item.stem
                        new_name = f"{stem}_{counter}{item.suffix}"
                        target_path = base_path / category / new_name
                        counter += 1
                    
                    try:
                        shutil.move(str(item), str(target_path))
                        moved_files.append((item.name, category))
                        moved = True
                        break
                    except Exception as e:
                        print(f"Error moving {item.name}: {e}")
                        skipped_files.append(item.name)
                        break
            
            # If file doesn't match any category, move to "Other"
            if not moved:
                other_path = base_path / "Other"
                other_path.mkdir(exist_ok=True)
                target_path = other_path / item.name
                
                # Handle naming conflicts for Other category
                counter = 1
                while target_path.exists():
                    stem = item.stem
                    new_name = f"{stem}_{counter}{item.suffix}"
                    target_path = other_path / new_name
                    counter += 1
                
                try:
                    shutil.move(str(item), str(target_path))
                    moved_files.append((item.name, "Other"))
                except Exception as e:
                    print(f"Error moving {item.name}: {e}")
                    skipped_files.append(item.name)
    
    # Print summary
    print(f"\nOrganization complete for: {base_path}")
    print(f"Total files moved: {len(moved_files)}")
    
    if moved_files:
        print("\nFiles moved:")
        for filename, category in moved_files:
            print(f"  {filename} -> {category}/")
    
    if skipped_files:
        print(f"\nFiles skipped (could not move): {len(skipped_files)}")
        for filename in skipped_files:
            print(f"  {filename}")
    
    # Remove empty category folders (except Other)
    for category in categories:
        category_path = base_path / category
        try:
            if category_path.exists() and not any(category_path.iterdir()):
                category_path.rmdir()
                print(f"Removed empty folder: {category}")
        except Exception as e:
            print(f"Could not remove folder {category}: {e}")

if __name__ == "__main__":
    # Get directory from user or use current directory
    import sys
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        target_dir = input("Enter directory to organize (press Enter for current): ").strip()
        if not target_dir:
            target_dir = "."
    
    # Validate directory
    if not Path(target_dir).exists():
        print(f"Error: Directory '{target_dir}' does not exist.")
        sys.exit(1)
    
    # Confirm with user
    confirm = input(f"Organize files in '{target_dir}'? (y/n): ").strip().lower()
    if confirm == 'y':
        organize_files(target_dir)
    else:
        print("Operation cancelled.")