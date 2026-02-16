
import sys
import os

def remove_duplicates(input_file, output_file=None):
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return False
    
    if output_file is None:
        output_file = input_file + ".deduped"
    
    seen_lines = set()
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
        
        unique_lines = []
        for line in lines:
            if line not in seen_lines:
                seen_lines.add(line)
                unique_lines.append(line)
        
        with open(output_file, 'w', encoding='utf-8') as outfile:
            outfile.writelines(unique_lines)
        
        print(f"Successfully removed duplicates. Original: {len(lines)} lines, Deduped: {len(unique_lines)} lines")
        print(f"Output saved to: {output_file}")
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
    
    remove_duplicates(input_file, output_file)