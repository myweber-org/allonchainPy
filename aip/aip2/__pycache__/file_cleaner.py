
import sys

def remove_duplicate_lines(input_file, output_file):
    seen = set()
    unique_lines = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.rstrip('\n')
                if stripped_line not in seen:
                    seen.add(stripped_line)
                    unique_lines.append(line)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(unique_lines)
            
        return True, f"Successfully removed duplicates. Original: {len(seen) + (len(unique_lines) - len(seen))}, Unique: {len(unique_lines)}"
    
    except FileNotFoundError:
        return False, f"Error: Input file '{input_file}' not found."
    except Exception as e:
        return False, f"Error: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python file_cleaner.py <input_file> <output_file>")
        sys.exit(1)
    
    success, message = remove_duplicate_lines(sys.argv[1], sys.argv[2])
    print(message)