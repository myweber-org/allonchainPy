import csv
import re
from typing import List, Dict, Optional

def clean_string(value: str) -> str:
    """Remove extra whitespace and normalize string."""
    if not isinstance(value, str):
        return str(value)
    cleaned = re.sub(r'\s+', ' ', value.strip())
    return cleaned

def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def read_csv_file(filepath: str) -> List[Dict]:
    """Read CSV file and return list of dictionaries."""
    data = []
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
    except Exception as e:
        print(f"Error reading CSV: {e}")
    return data

def clean_csv_data(data: List[Dict], columns_to_clean: Optional[List[str]] = None) -> List[Dict]:
    """Clean specified columns in CSV data."""
    cleaned_data = []
    for row in data:
        cleaned_row = {}
        for key, value in row.items():
            if columns_to_clean is None or key in columns_to_clean:
                if isinstance(value, str):
                    cleaned_row[key] = clean_string(value)
                else:
                    cleaned_row[key] = value
            else:
                cleaned_row[key] = value
        cleaned_data.append(cleaned_row)
    return cleaned_data

def write_csv_file(data: List[Dict], filepath: str) -> bool:
    """Write data to CSV file."""
    if not data:
        return False
    
    try:
        fieldnames = data[0].keys()
        with open(filepath, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        return True
    except Exception as e:
        print(f"Error writing CSV: {e}")
        return False

def filter_invalid_emails(data: List[Dict], email_column: str) -> List[Dict]:
    """Filter rows with invalid email addresses."""
    valid_data = []
    for row in data:
        if email_column in row and validate_email(row[email_column]):
            valid_data.append(row)
    return valid_data

def process_csv_pipeline(input_file: str, output_file: str, email_column: str = 'email') -> None:
    """Complete CSV processing pipeline."""
    print(f"Processing {input_file}...")
    
    raw_data = read_csv_file(input_file)
    if not raw_data:
        print("No data to process.")
        return
    
    print(f"Read {len(raw_data)} rows.")
    
    cleaned_data = clean_csv_data(raw_data)
    valid_data = filter_invalid_emails(cleaned_data, email_column)
    
    print(f"Filtered to {len(valid_data)} valid rows.")
    
    if write_csv_file(valid_data, output_file):
        print(f"Successfully wrote cleaned data to {output_file}")
    else:
        print("Failed to write output file.")

if __name__ == "__main__":
    process_csv_pipeline('input.csv', 'output.csv')