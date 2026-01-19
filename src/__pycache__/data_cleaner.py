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
        print(f"Input file '{input_file}' does not exist.")import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data.copy()
    
    normalized = data.copy()
    normalized[column] = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        DataFrame with standardized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data.copy()
    
    standardized = data.copy()
    standardized[column] = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns=None):
    """
    Apply cleaning operations to a dataset.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for col in numeric_columns:
        if col in cleaned_data.columns:
            cleaned_data = remove_outliers_iqr(cleaned_data, col)
            cleaned_data = standardize_zscore(cleaned_data, col)
    
    return cleaned_data

def validate_data(data, required_columns, allow_nan=False):
    """
    Validate dataset structure and content.
    
    Args:
        data: pandas DataFrame to validate
        required_columns: list of required column names
        allow_nan: whether to allow NaN values
    
    Returns:
        tuple: (is_valid, error_message)
    """
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"
    
    if not allow_nan:
        nan_columns = data.columns[data.isnull().any()].tolist()
        if nan_columns:
            return False, f"NaN values found in columns: {nan_columns}"
    
    return True, "Data validation passed"