import re
import pandas as pd
from typing import Union, List, Optional

def validate_email(email: str) -> bool:
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def clean_phone_number(phone: str) -> Optional[str]:
    digits = re.sub(r'\D', '', phone)
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    elif len(digits) == 11 and digits.startswith('1'):
        return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
    return None

def remove_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.DataFrame:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [col.lower().replace(' ', '_').strip() for col in df.columns]
    return df

def fill_missing_with_median(df: pd.DataFrame, columns: Union[str, List[str]]) -> pd.DataFrame:
    if isinstance(columns, str):
        columns = [columns]
    for col in columns:
        if col in df.columns:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
    return df
import pandas as pd
import numpy as np
from typing import List, Optional

def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop', fill_value: Optional[float] = None) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: 'drop' to remove rows, 'fill' to replace with value
        fill_value: Value to use when strategy is 'fill'
    
    Returns:
        DataFrame with handled missing values
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill':
        if fill_value is not None:
            return df.fillna(fill_value)
        else:
            return df.fillna(df.mean(numeric_only=True))
    else:
        raise ValueError("Strategy must be 'drop' or 'fill'")

def normalize_numeric_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Normalize specified numeric columns to 0-1 range.
    
    Args:
        df: Input DataFrame
        columns: List of column names to normalize
    
    Returns:
        DataFrame with normalized columns
    """
    result = df.copy()
    for col in columns:
        if col in result.columns and pd.api.types.is_numeric_dtype(result[col]):
            col_min = result[col].min()
            col_max = result[col].max()
            if col_max > col_min:
                result[col] = (result[col] - col_min) / (col_max - col_min)
    return result

def clean_dataset(df: pd.DataFrame, 
                  deduplicate: bool = True,
                  missing_strategy: str = 'drop',
                  normalize_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        deduplicate: Whether to remove duplicates
        missing_strategy: Strategy for handling missing values
        normalize_cols: Columns to normalize
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if deduplicate:
        cleaned_df = remove_duplicates(cleaned_df)
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    if normalize_cols:
        cleaned_df = normalize_numeric_columns(cleaned_df, normalize_cols)
    
    return cleaned_df
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def normalize_minmax(df, columns):
    df_norm = df.copy()
    for col in columns:
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
    return df_norm

def clean_dataset(file_path, numeric_columns):
    df = pd.read_csv(file_path)
    df_clean = remove_outliers_iqr(df, numeric_columns)
    df_normalized = normalize_minmax(df_clean, numeric_columns)
    return df_normalized

if __name__ == "__main__":
    cleaned_data = clean_dataset('sample_data.csv', ['age', 'income', 'score'])
    cleaned_data.to_csv('cleaned_data.csv', index=False)
    print(f"Data cleaning complete. Original shape: {pd.read_csv('sample_data.csv').shape}, Cleaned shape: {cleaned_data.shape}")import csv
import os
from typing import List, Dict, Any, Optional

def read_csv_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Read a CSV file and return its contents as a list of dictionaries.
    """
    data = []
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return data
    
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
    
    return data

def clean_numeric_column(data: List[Dict[str, Any]], column_name: str) -> List[Dict[str, Any]]:
    """
    Clean a numeric column by removing non-numeric characters and converting to float.
    """
    cleaned_data = []
    for row in data:
        cleaned_row = row.copy()
        if column_name in cleaned_row:
            value = cleaned_row[column_name]
            if isinstance(value, str):
                # Remove non-numeric characters except decimal point and minus sign
                cleaned_value = ''.join(ch for ch in value if ch.isdigit() or ch in '.-')
                try:
                    cleaned_row[column_name] = float(cleaned_value) if cleaned_value else 0.0
                except ValueError:
                    cleaned_row[column_name] = 0.0
        cleaned_data.append(cleaned_row)
    return cleaned_data

def remove_empty_rows(data: List[Dict[str, Any]], key_columns: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Remove rows where all specified key columns are empty.
    If key_columns is None, check all columns.
    """
    if not data:
        return []
    
    if key_columns is None:
        key_columns = list(data[0].keys())
    
    filtered_data = []
    for row in data:
        # Check if all key columns are empty
        if not all(not row.get(col) for col in key_columns):
            filtered_data.append(row)
    
    return filtered_data

def write_csv_file(data: List[Dict[str, Any]], file_path: str) -> bool:
    """
    Write data to a CSV file.
    """
    if not data:
        print("Error: No data to write.")
        return False
    
    try:
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = list(data[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        return True
    except Exception as e:
        print(f"Error writing CSV file: {e}")
        return False

def get_column_stats(data: List[Dict[str, Any]], column_name: str) -> Dict[str, Any]:
    """
    Calculate basic statistics for a numeric column.
    """
    values = []
    for row in data:
        if column_name in row and isinstance(row[column_name], (int, float)):
            values.append(row[column_name])
    
    if not values:
        return {"count": 0, "min": None, "max": None, "avg": None}
    
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "avg": sum(values) / len(values)
    }import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to drop duplicate rows
        fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if cleaned_df.isnull().sum().sum() > 0:
        print(f"Found {cleaned_df.isnull().sum().sum()} missing values")
        
        if fill_missing == 'drop':
            cleaned_df = cleaned_df.dropna()
            print("Dropped rows with missing values")
        elif fill_missing in ['mean', 'median']:
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if fill_missing == 'mean':
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                else:
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            print(f"Filled missing numeric values with {fill_missing}")
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 0)
            print("Filled missing values with mode")
    
    print(f"Cleaning complete. Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'is_dataframe': isinstance(df, pd.DataFrame),
        'has_data': len(df) > 0,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict()
    }
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_cols
        validation_results['all_required_present'] = len(missing_cols) == 0
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 3, None, 5],
        'B': [10, None, 10, 20, 30, 40],
        'C': ['x', 'y', 'x', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    validation = validate_dataframe(cleaned, required_columns=['A', 'B'])
    print("\nValidation Results:")
    for key, value in validation.items():
        print(f"{key}: {value}")