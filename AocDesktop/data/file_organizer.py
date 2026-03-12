
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organizes files in the given directory by moving them into
    subfolders based on their file extensions.
    """
    # Define categories and their associated extensions
    categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Archives': ['.zip', '.tar', '.gz', '.rar', '.7z'],
        'Audio': ['.mp3', '.wav', '.flac', '.aac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv', '.wmv'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c']
    }

    # Ensure the directory exists
    target_dir = Path(directory)
    if not target_dir.exists() or not target_dir.is_dir():
        print(f"Error: Directory '{directory}' does not exist.")
        return

    # Create category folders if they don't exist
    for category in categories.keys():
        category_path = target_dir / category
        category_path.mkdir(exist_ok=True)

    # Track moved files and unknown extensions
    moved_files = []
    unknown_extensions = set()

    # Iterate over all items in the directory
    for item in target_dir.iterdir():
        # Skip directories and hidden files
        if item.is_dir() or item.name.startswith('.'):
            continue

        # Get file extension
        extension = item.suffix.lower()

        # Find the appropriate category
        destination_category = None
        for category, extensions in categories.items():
            if extension in extensions:
                destination_category = category
                break

        # Move file to the category folder
        if destination_category:
            destination_path = target_dir / destination_category / item.name
            # Handle duplicate filenames
            if destination_path.exists():
                base_name = item.stem
                counter = 1
                while destination_path.exists():
                    new_name = f"{base_name}_{counter}{item.suffix}"
                    destination_path = target_dir / destination_category / new_name
                    counter += 1

            try:
                shutil.move(str(item), str(destination_path))
                moved_files.append((item.name, destination_category))
            except Exception as e:
                print(f"Error moving file {item.name}: {e}")
        else:
            unknown_extensions.add(extension)

    # Print summary
    print(f"Organization complete for '{directory}'")
    if moved_files:
        print(f"Moved {len(moved_files)} files:")
        for filename, category in moved_files:
            print(f"  {filename} -> {category}/")
    else:
        print("No files were moved.")

    if unknown_extensions:
        print(f"Files with unknown extensions were not moved: {', '.join(unknown_extensions)}")

if __name__ == "__main__":
    # Example usage: organize files in the current directory
    organize_files('.')
import os
import shutil
from pathlib import Path

def organize_files(directory="."):
    """
    Organizes files in the specified directory by moving them into
    subdirectories based on their file extensions.
    """
    base_path = Path(directory)
    
    # Define categories and their associated extensions
    categories = {
        "Images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg"],
        "Documents": [".pdf", ".docx", ".txt", ".xlsx", ".pptx", ".md"],
        "Audio": [".mp3", ".wav", ".flac", ".aac"],
        "Video": [".mp4", ".avi", ".mov", ".mkv"],
        "Archives": [".zip", ".tar", ".gz", ".7z", ".rar"],
        "Code": [".py", ".js", ".html", ".css", ".java", ".cpp"],
        "Executables": [".exe", ".msi", ".sh", ".bat"]
    }
    
    # Create category folders if they don't exist
    for category in categories:
        (base_path / category).mkdir(exist_ok=True)
    
    # Create 'Others' folder for uncategorized files
    others_folder = base_path / "Others"
    others_folder.mkdir(exist_ok=True)
    
    # Track moved files
    moved_files = []
    skipped_files = []
    
    for item in base_path.iterdir():
        if item.is_file():
            file_moved = False
            file_ext = item.suffix.lower()
            
            # Find the appropriate category
            for category, extensions in categories.items():
                if file_ext in extensions:
                    dest_folder = base_path / category
                    try:
                        shutil.move(str(item), str(dest_folder / item.name))
                        moved_files.append((item.name, category))
                        file_moved = True
                        break
                    except Exception as e:
                        print(f"Error moving {item.name}: {e}")
                        skipped_files.append(item.name)
                        file_moved = True
                        break
            
            # If file doesn't match any category, move to 'Others'
            if not file_moved:
                try:
                    shutil.move(str(item), str(others_folder / item.name))
                    moved_files.append((item.name, "Others"))
                except Exception as e:
                    print(f"Error moving {item.name}: {e}")
                    skipped_files.append(item.name)
    
    # Print summary
    print(f"\nOrganization complete!")
    print(f"Total files processed: {len(moved_files) + len(skipped_files)}")
    print(f"Files successfully moved: {len(moved_files)}")
    print(f"Files skipped/errors: {len(skipped_files)}")
    
    if moved_files:
        print("\nMoved files:")
        for filename, category in moved_files:
            print(f"  {filename} -> {category}/")
    
    if skipped_files:
        print("\nSkipped files:")
        for filename in skipped_files:
            print(f"  {filename}")

if __name__ == "__main__":
    # Get directory from user or use current directory
    target_dir = input("Enter directory to organize (press Enter for current): ").strip()
    if not target_dir:
        target_dir = "."
    
    if Path(target_dir).exists():
        organize_files(target_dir)
    else:
        print(f"Error: Directory '{target_dir}' does not exist.")