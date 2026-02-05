
import pandas as pd
import numpy as np

def remove_outliers(df, column, threshold=3):
    mean = df[column].mean()
    std = df[column].std()
    z_scores = np.abs((df[column] - mean) / std)
    return df[z_scores < threshold]

def normalize_column(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path, output_path):
    df = pd.read_csv(file_path)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers(df, col)
        df = normalize_column(df, col)
    
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    cleaned_df = clean_dataset(input_file, output_file)
    print(f"Data cleaning completed. Cleaned data saved to {output_file}")
import csv
import sys

def remove_duplicates(input_file, output_file, key_column):
    """
    Remove duplicate rows from a CSV file based on a specified key column.
    """
    seen = set()
    cleaned_rows = []
    
    try:
        with open(input_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames
            
            for row in reader:
                key = row.get(key_column)
                if key not in seen:
                    seen.add(key)
                    cleaned_rows.append(row)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(cleaned_rows)
        
        print(f"Cleaned data saved to {output_file}")
        print(f"Removed {len(seen) - len(cleaned_rows)} duplicate rows")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except KeyError:
        print(f"Error: Key column '{key_column}' not found in CSV.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python data_cleaner.py <input_file> <output_file> <key_column>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    key_column = sys.argv[3]
    
    remove_duplicates(input_file, output_file, key_column)import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=False, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default True.
    fill_missing (bool): Whether to fill missing values. Default False.
    fill_value: Value to use for filling missing data. Default 0.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    return True, "Data validation passed"

def normalize_column(df, column_name):
    """
    Normalize a numeric column to range [0, 1].
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column_name (str): Name of column to normalize.
    
    Returns:
    pd.DataFrame: DataFrame with normalized column.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        raise ValueError(f"Column '{column_name}' is not numeric")
    
    col_min = df[column_name].min()
    col_max = df[column_name].max()
    
    if col_max == col_min:
        df[f"{column_name}_normalized"] = 0.5
    else:
        df[f"{column_name}_normalized"] = (df[column_name] - col_min) / (col_max - col_min)
    
    return df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'value': [10, 20, 20, None, 40],
        'category': ['A', 'B', 'B', 'C', 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned = clean_dataset(df, fill_missing=True, fill_value=0)
    print("Cleaned DataFrame:")
    print(cleaned)
    print()
    
    is_valid, message = validate_data(cleaned, required_columns=['id', 'value'])
    print(f"Validation: {is_valid} - {message}")
    print()
    
    normalized = normalize_column(cleaned, 'value')
    print("DataFrame with normalized 'value' column:")
    print(normalized)import pandas as pd
import numpy as np
from typing import List, Union

def remove_duplicates(df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def convert_column_types(df: pd.DataFrame, 
                         column_types: dict) -> pd.DataFrame:
    """
    Convert columns to specified data types.
    
    Args:
        df: Input DataFrame
        column_types: Dictionary mapping column names to target types
    
    Returns:
        DataFrame with converted column types
    """
    df_copy = df.copy()
    for column, dtype in column_types.items():
        if column in df_copy.columns:
            try:
                df_copy[column] = df_copy[column].astype(dtype)
            except (ValueError, TypeError):
                df_copy[column] = pd.to_numeric(df_copy[column], errors='coerce')
    return df_copy

def handle_missing_values(df: pd.DataFrame, 
                         strategy: str = 'mean',
                         columns: List[str] = None) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: Imputation strategy ('mean', 'median', 'mode', 'drop')
        columns: Specific columns to process
    
    Returns:
        DataFrame with handled missing values
    """
    df_copy = df.copy()
    
    if columns is None:
        columns = df_copy.columns
    
    for column in columns:
        if column not in df_copy.columns:
            continue
            
        if strategy == 'drop':
            df_copy = df_copy.dropna(subset=[column])
        elif strategy == 'mean':
            df_copy[column] = df_copy[column].fillna(df_copy[column].mean())
        elif strategy == 'median':
            df_copy[column] = df_copy[column].fillna(df_copy[column].median())
        elif strategy == 'mode':
            df_copy[column] = df_copy[column].fillna(df_copy[column].mode()[0])
    
    return df_copy

def normalize_data(df: pd.DataFrame,
                   columns: List[str],
                   method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize specified columns in DataFrame.
    
    Args:
        df: Input DataFrame
        columns: Columns to normalize
        method: Normalization method ('minmax' or 'zscore')
    
    Returns:
        DataFrame with normalized columns
    """
    df_copy = df.copy()
    
    for column in columns:
        if column not in df_copy.columns:
            continue
            
        if method == 'minmax':
            min_val = df_copy[column].min()
            max_val = df_copy[column].max()
            if max_val != min_val:
                df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            mean_val = df_copy[column].mean()
            std_val = df_copy[column].std()
            if std_val != 0:
                df_copy[column] = (df_copy[column] - mean_val) / std_val
    
    return df_copy

def clean_dataframe(df: pd.DataFrame,
                   deduplicate: bool = True,
                   type_conversions: dict = None,
                   missing_strategy: str = 'mean',
                   normalize_cols: List[str] = None) -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        deduplicate: Whether to remove duplicates
        type_conversions: Dictionary for column type conversions
        missing_strategy: Strategy for handling missing values
        normalize_cols: Columns to normalize
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if deduplicate:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if type_conversions:
        cleaned_df = convert_column_types(cleaned_df, type_conversions)
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    if normalize_cols:
        cleaned_df = normalize_data(cleaned_df, normalize_cols)
    
    return cleaned_df
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name
    
    Returns:
    dict: Dictionary containing statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def main():
    # Example usage
    data = {
        'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 102, 12, 14, 13, 12, 11, 14, 13, 12, 14, 100]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print(f"\nOriginal shape: {df.shape}")
    
    # Remove outliers
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print(f"\nCleaned shape: {cleaned_df.shape}")
    
    # Calculate statistics
    original_stats = calculate_statistics(df, 'values')
    cleaned_stats = calculate_statistics(cleaned_df, 'values')
    
    print("\nOriginal statistics:")
    for key, value in original_stats.items():
        print(f"{key}: {value:.2f}")
    
    print("\nCleaned statistics:")
    for key, value in cleaned_stats.items():
        print(f"{key}: {value:.2f}")

if __name__ == "__main__":
    main()
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_data(input_list):
    if not isinstance(input_list, list):
        raise TypeError("Input must be a list")
    cleaned = remove_duplicates_preserve_order(input_list)
    return cleaned

if __name__ == "__main__":
    sample_data = [3, 1, 2, 3, 4, 2, 5, 1, 6]
    print("Original:", sample_data)
    print("Cleaned:", clean_data(sample_data))