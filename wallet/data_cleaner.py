import csv
import os

def clean_csv(input_path, output_path, columns_to_keep=None, delimiter=','):
    """
    Clean a CSV file by removing rows with missing values in specified columns.
    If columns_to_keep is provided, only those columns are retained.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    cleaned_rows = []
    with open(input_path, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile, delimiter=delimiter)
        fieldnames = reader.fieldnames
        
        if columns_to_keep:
            # Validate that specified columns exist
            for col in columns_to_keep:
                if col not in fieldnames:
                    raise ValueError(f"Column '{col}' not found in CSV header")
            fieldnames = columns_to_keep
        
        cleaned_rows.append(fieldnames)
        
        for row in reader:
            # Skip rows with any missing values in columns_to_keep (or all columns if None)
            columns_to_check = columns_to_keep if columns_to_keep else fieldnames
            if any(row.get(col, '').strip() == '' for col in columns_to_check):
                continue
            
            # Build cleaned row
            cleaned_row = [row[col] for col in fieldnames]
            cleaned_rows.append(cleaned_row)
    
    # Write cleaned data
    with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile, delimiter=delimiter)
        writer.writerows(cleaned_rows)
    
    return len(cleaned_rows) - 1  # Return number of cleaned rows (excluding header)

def validate_csv(file_path, required_columns=None):
    """
    Validate CSV file structure and required columns.
    Returns True if valid, raises ValueError otherwise.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', newline='', encoding='utf-8') as f:
        try:
            reader = csv.reader(f)
            header = next(reader)
        except (csv.Error, StopIteration):
            raise ValueError("Invalid CSV format or empty file")
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in header]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

def get_csv_stats(file_path):
    """
    Get basic statistics about a CSV file.
    """
    validate_csv(file_path)
    
    with open(file_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    
    return {
        'file': os.path.basename(file_path),
        'columns': len(header),
        'column_names': header,
        'total_rows': len(rows),
        'non_empty_rows': sum(1 for row in rows if any(cell.strip() for cell in row))
    }