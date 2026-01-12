
import csv
import sys
from pathlib import Path

def remove_duplicates(input_file, output_file=None, key_column=None):
    """
    Remove duplicate rows from a CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str, optional): Path for output CSV file. 
                                     If None, creates input_file_cleaned.csv
        key_column (str, optional): Column name to identify duplicates.
                                    If None, uses entire row for comparison.
    
    Returns:
        int: Number of duplicate rows removed
    """
    if not Path(input_file).exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_cleaned.csv"
    
    seen_rows = set()
    unique_rows = []
    duplicates_removed = 0
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        for row in reader:
            if key_column:
                key = row.get(key_column)
                if key in seen_rows:
                    duplicates_removed += 1
                    continue
                seen_rows.add(key)
            else:
                row_tuple = tuple(row.items())
                if row_tuple in seen_rows:
                    duplicates_removed += 1
                    continue
                seen_rows.add(row_tuple)
            
            unique_rows.append(row)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unique_rows)
    
    return duplicates_removed

def main():
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_file> [output_file] [key_column]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    key_column = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        removed = remove_duplicates(input_file, output_file, key_column)
        print(f"Successfully removed {removed} duplicate rows")
        print(f"Cleaned data saved to: {output_file if output_file else 'input_file_cleaned.csv'}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()