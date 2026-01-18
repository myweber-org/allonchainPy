import csv
import re
from typing import List, Dict, Any

def clean_csv_data(input_file: str, output_file: str, columns_to_clean: List[str]) -> None:
    """
    Clean specified columns in a CSV file by removing extra whitespace
    and converting strings to lowercase.
    """
    cleaned_rows = []
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        for row in reader:
            cleaned_row = {}
            for key, value in row.items():
                if key in columns_to_clean and isinstance(value, str):
                    # Remove extra whitespace and convert to lowercase
                    cleaned_value = re.sub(r'\s+', ' ', value.strip()).lower()
                    cleaned_row[key] = cleaned_value
                else:
                    cleaned_row[key] = value
            cleaned_rows.append(cleaned_row)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)

def validate_email_column(input_file: str, email_column: str) -> List[Dict[str, Any]]:
    """
    Validate email addresses in a specified column and return rows with invalid emails.
    """
    invalid_rows = []
    email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        
        for row_number, row in enumerate(reader, start=2):  # Start at 2 for header row
            email = row.get(email_column, '')
            if email and not email_pattern.match(email):
                invalid_rows.append({
                    'row_number': row_number,
                    'data': row,
                    'invalid_email': email
                })
    
    return invalid_rows

def remove_duplicate_rows(input_file: str, output_file: str, key_columns: List[str]) -> None:
    """
    Remove duplicate rows based on specified key columns.
    """
    seen = set()
    unique_rows = []
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        for row in reader:
            # Create a tuple of values from key columns for comparison
            key_tuple = tuple(row[col] for col in key_columns)
            
            if key_tuple not in seen:
                seen.add(key_tuple)
                unique_rows.append(row)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unique_rows)