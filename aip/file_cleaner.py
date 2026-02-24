import sys

def remove_duplicates(input_file, output_file):
    seen = set()
    unique_lines = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                stripped = line.rstrip('\n')
                if stripped not in seen:
                    seen.add(stripped)
                    unique_lines.append(line)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return False
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(unique_lines)
        print(f"Successfully removed duplicates. Output saved to '{output_file}'.")
        return True
    except IOError as e:
        print(f"Error writing to file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python file_cleaner.py <input_file> <output_file>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    remove_duplicates(input_path, output_path)