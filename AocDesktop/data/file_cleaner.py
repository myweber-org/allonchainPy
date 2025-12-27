import sys

def remove_duplicates(input_file, output_file=None):
    """
    Remove duplicate lines from a file while preserving order.
    """
    if output_file is None:
        output_file = input_file + '.deduped'
    
    seen = set()
    unique_lines = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line not in seen:
                    seen.add(line)
                    unique_lines.append(line)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return False
    except IOError as e:
        print(f"Error reading file: {e}")
        return False
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(unique_lines)
    except IOError as e:
        print(f"Error writing file: {e}")
        return False
    
    print(f"Removed {len(seen) - len(unique_lines)} duplicate lines.")
    print(f"Unique lines saved to: {output_file}")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_cleaner.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = remove_duplicates(input_file, output_file)
    sys.exit(0 if success else 1)