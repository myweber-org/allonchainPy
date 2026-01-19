import csv
import os
from typing import List, Dict, Any, Optional

def read_csv(file_path: str) -> List[Dict[str, Any]]:
    """Read a CSV file and return a list of dictionaries."""
    data = []
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
    return data

def write_csv(file_path: str, data: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> bool:
    """Write data to a CSV file."""
    if not data:
        print("No data to write.")
        return False
    
    if fieldnames is None:
        fieldnames = list(data[0].keys())
    
    try:
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        return True
    except Exception as e:
        print(f"Error writing CSV file: {e}")
        return False

def remove_empty_rows(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove rows where all values are empty strings."""
    cleaned_data = []
    for row in data:
        if any(value.strip() != '' for value in row.values()):
            cleaned_data.append(row)
    return cleaned_data

def fill_missing_values(data: List[Dict[str, Any]], default_value: str = "N/A") -> List[Dict[str, Any]]:
    """Replace empty string values with a default value."""
    for row in data:
        for key in row:
            if row[key] == '':
                row[key] = default_value
    return data

def normalize_column_names(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize column names to lowercase with underscores."""
    if not data:
        return data
    
    normalized_data = []
    for row in data:
        new_row = {}
        for key, value in row.items():
            new_key = key.lower().replace(' ', '_')
            new_row[new_key] = value
        normalized_data.append(new_row)
    return normalized_data

def clean_csv(input_path: str, output_path: str) -> None:
    """Apply cleaning operations to a CSV file."""
    print(f"Cleaning data from '{input_path}'...")
    
    data = read_csv(input_path)
    if not data:
        print("No data loaded. Exiting.")
        return
    
    print(f"Loaded {len(data)} rows.")
    
    data = remove_empty_rows(data)
    print(f"After removing empty rows: {len(data)} rows.")
    
    data = fill_missing_values(data)
    data = normalize_column_names(data)
    
    if write_csv(output_path, data):
        print(f"Cleaned data saved to '{output_path}'.")
    else:
        print("Failed to save cleaned data.")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    if os.path.exists(input_file):
        clean_csv(input_file, output_file)
    else:
        print(f"Input file '{input_file}' does not exist.")