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
        writer.writerows(unique_rows)import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Args:
        data (pd.DataFrame): Input dataframe
        column (str): Column name to clean
    
    Returns:
        pd.DataFrame: Dataframe with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    return filtered_data.reset_index(drop=True)

def calculate_basic_stats(data, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Args:
        data (pd.DataFrame): Input dataframe
        column (str): Column name to analyze
    
    Returns:
        dict: Dictionary containing statistics
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    stats = {
        'mean': np.mean(data[column]),
        'median': np.median(data[column]),
        'std': np.std(data[column]),
        'min': np.min(data[column]),
        'max': np.max(data[column]),
        'count': len(data[column])
    }
    
    return stats

def clean_dataset(data, columns_to_clean):
    """
    Clean multiple columns in a dataset by removing outliers.
    
    Args:
        data (pd.DataFrame): Input dataframe
        columns_to_clean (list): List of column names to clean
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    cleaned_data = data.copy()
    
    for column in columns_to_clean:
        if column in cleaned_data.columns:
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
    
    return cleaned_data