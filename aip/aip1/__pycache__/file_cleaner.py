
import sys
import os

def remove_duplicates(input_file, output_file=None):
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return False
    
    if output_file is None:
        output_file = input_file + ".deduped"
    
    seen_lines = set()
    lines_removed = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
        
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for line in lines:
                if line not in seen_lines:
                    outfile.write(line)
                    seen_lines.add(line)
                else:
                    lines_removed += 1
        
        print(f"Successfully processed '{input_file}'")
        print(f"Removed {lines_removed} duplicate lines")
        print(f"Output saved to '{output_file}'")
        return True
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_cleaner.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = remove_duplicates(input_file, output_file)
    sys.exit(0 if success else 1)