import pandas as pd
import numpy as np

def clean_csv_data(file_path, missing_strategy='drop', fill_value=0):
    """
    Load and clean CSV data.
    
    Args:
        file_path (str): Path to CSV file.
        missing_strategy (str): Strategy for handling missing values.
            'drop': Drop rows with missing values.
            'fill': Fill missing values with specified fill_value.
        fill_value: Value to fill missing data if strategy is 'fill'.
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if missing_strategy == 'drop':
        df_cleaned = df.dropna()
    elif missing_strategy == 'fill':
        df_cleaned = df.fillna(fill_value)
    else:
        raise ValueError("Invalid missing_strategy. Use 'drop' or 'fill'.")
    
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    df_cleaned[numeric_cols] = df_cleaned[numeric_cols].apply(
        lambda x: pd.to_numeric(x, errors='coerce')
    )
    
    return df_cleaned

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df (pandas.DataFrame): Input DataFrame.
        subset (list): Columns to consider for duplicates.
    
    Returns:
        pandas.DataFrame: DataFrame with duplicates removed.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def normalize_numeric_columns(df, columns=None):
    """
    Normalize numeric columns to range [0, 1].
    
    Args:
        df (pandas.DataFrame): Input DataFrame.
        columns (list): Columns to normalize. If None, normalize all numeric columns.
    
    Returns:
        pandas.DataFrame: DataFrame with normalized columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max != col_min:
                df_normalized[col] = (df[col] - col_min) / (col_max - col_min)
    
    return df_normalized

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to CSV.
    
    Args:
        df (pandas.DataFrame): DataFrame to save.
        output_path (str): Path for output CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")import csv
import re
from typing import List, Dict, Any, Optional

def clean_string(value: str) -> str:
    """Remove extra whitespace and normalize string."""
    if not isinstance(value, str):
        return str(value)
    cleaned = re.sub(r'\s+', ' ', value.strip())
    return cleaned

def parse_numeric(value: str) -> Optional[float]:
    """Attempt to parse string as float, handling common issues."""
    if value is None:
        return None
    cleaned = value.replace(',', '').strip()
    if cleaned == '':
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None

def validate_email(email: str) -> bool:
    """Basic email validation using regex."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email.strip()))

def clean_csv_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Apply cleaning functions to all values in a CSV row dictionary."""
    cleaned_row = {}
    for key, value in row.items():
        if isinstance(value, str):
            cleaned_row[key] = clean_string(value)
        else:
            cleaned_row[key] = value
    return cleaned_row

def read_and_clean_csv(filepath: str) -> List[Dict[str, Any]]:
    """Read CSV file and clean all rows."""
    cleaned_data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            cleaned_row = clean_csv_row(row)
            cleaned_data.append(cleaned_row)
    return cleaned_data

def write_cleaned_csv(data: List[Dict[str, Any]], output_path: str) -> None:
    """Write cleaned data to a new CSV file."""
    if not data:
        return
    fieldnames = data[0].keys()
    with open(output_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def remove_duplicates(data: List[Dict[str, Any]], key_field: str) -> List[Dict[str, Any]]:
    """Remove duplicate rows based on a specified key field."""
    seen = set()
    unique_data = []
    for row in data:
        key_value = row.get(key_field)
        if key_value not in seen:
            seen.add(key_value)
            unique_data.append(row)
    return unique_data

if __name__ == "__main__":
    sample_data = [
        {"id": "1", "name": "  John   Doe  ", "email": "john@example.com", "score": "95.5"},
        {"id": "2", "name": "Jane Smith", "email": "invalid-email", "score": "88.0"},
        {"id": "3", "name": "Bob   Johnson", "email": "bob@test.org", "score": "N/A"}
    ]
    
    cleaned = [clean_csv_row(row) for row in sample_data]
    print("Cleaned sample data:")
    for row in cleaned:
        print(row)