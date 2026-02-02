import csv
import re
from typing import List, Dict, Any

def remove_duplicates(data: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    seen = set()
    unique_data = []
    for row in data:
        if row[key] not in seen:
            seen.add(row[key])
            unique_data.append(row)
    return unique_data

def normalize_string(value: str) -> str:
    if not isinstance(value, str):
        return value
    value = value.strip().lower()
    value = re.sub(r'\s+', ' ', value)
    return value

def clean_numeric(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = re.sub(r'[^\d.-]', '', value)
        try:
            return float(cleaned) if cleaned else 0.0
        except ValueError:
            return 0.0
    return 0.0

def validate_email(email: str) -> bool:
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def process_csv_file(input_path: str, output_path: str, key_column: str) -> None:
    with open(input_path, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        data = list(reader)
    
    cleaned_data = remove_duplicates(data, key_column)
    
    for row in cleaned_data:
        for field in row:
            if field.endswith('_str'):
                row[field] = normalize_string(row[field])
            elif field.endswith('_num'):
                row[field] = clean_numeric(row[field])
            elif field == 'email':
                if not validate_email(row[field]):
                    row[field] = ''
    
    fieldnames = cleaned_data[0].keys() if cleaned_data else []
    with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_data)