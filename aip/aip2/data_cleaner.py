
import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file, missing_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to save cleaned CSV file
        missing_strategy (str): Strategy for handling missing values
                               ('mean', 'median', 'drop', 'zero')
    """
    try:
        df = pd.read_csv(input_file)
        
        print(f"Original data shape: {df.shape}")
        print(f"Missing values per column:\n{df.isnull().sum()}")
        
        df_cleaned = df.copy()
        
        if missing_strategy == 'drop':
            df_cleaned = df_cleaned.dropna()
        elif missing_strategy in ['mean', 'median']:
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if missing_strategy == 'mean':
                    fill_value = df_cleaned[col].mean()
                else:
                    fill_value = df_cleaned[col].median()
                df_cleaned[col] = df_cleaned[col].fillna(fill_value)
        elif missing_strategy == 'zero':
            df_cleaned = df_cleaned.fillna(0)
        
        df_cleaned = df_cleaned.drop_duplicates()
        
        df_cleaned.to_csv(output_file, index=False)
        
        print(f"Cleaned data shape: {df_cleaned.shape}")
        print(f"Cleaned data saved to: {output_file}")
        
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Args:
        df (pd.DataFrame): Dataframe to validate
        required_columns (list): List of required column names
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if df is None or df.empty:
        print("Error: Dataframe is empty or None")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    if df.isnull().any().any():
        print("Warning: Dataframe contains missing values")
    
    return True

if __name__ == "__main__":
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_csv, output_csv, missing_strategy='mean')
    
    if cleaned_df is not None:
        is_valid = validate_dataframe(cleaned_df)
        if is_valid:
            print("Data cleaning completed successfully")
        else:
            print("Data validation failed")
import pandas as pd
import numpy as np
from typing import List, Optional

def clean_column(df: pd.DataFrame, column_name: str, 
                 remove_duplicates: bool = True,
                 case_sensitive: bool = False,
                 strip_whitespace: bool = True) -> pd.DataFrame:
    """
    Clean a specified column in a DataFrame.
    
    Parameters:
    df: Input DataFrame
    column_name: Name of column to clean
    remove_duplicates: Remove duplicate values
    case_sensitive: Case sensitivity for duplicate detection
    strip_whitespace: Remove leading/trailing whitespace
    
    Returns:
    Cleaned DataFrame
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df_clean = df.copy()
    
    if strip_whitespace:
        df_clean[column_name] = df_clean[column_name].astype(str).str.strip()
    
    if not case_sensitive:
        df_clean[column_name] = df_clean[column_name].str.lower()
    
    if remove_duplicates:
        df_clean = df_clean.drop_duplicates(subset=[column_name], keep='first')
    
    return df_clean

def normalize_numeric(df: pd.DataFrame, column_name: str,
                      scale_range: Optional[tuple] = None) -> pd.DataFrame:
    """
    Normalize numeric column to specified range or 0-1.
    
    Parameters:
    df: Input DataFrame
    column_name: Numeric column to normalize
    scale_range: Target range as (min, max), defaults to (0, 1)
    
    Returns:
    DataFrame with normalized column
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        raise TypeError(f"Column '{column_name}' must be numeric")
    
    df_norm = df.copy()
    col_data = df_norm[column_name]
    
    if scale_range is None:
        scale_range = (0, 1)
    
    min_val, max_val = scale_range
    col_min = col_data.min()
    col_max = col_data.max()
    
    if col_max == col_min:
        df_norm[f"{column_name}_normalized"] = min_val
    else:
        normalized = min_val + (col_data - col_min) * (max_val - min_val) / (col_max - col_min)
        df_norm[f"{column_name}_normalized"] = normalized
    
    return df_norm

def validate_dataframe(df: pd.DataFrame, 
                       required_columns: List[str] = None) -> dict:
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df: DataFrame to validate
    required_columns: List of required column names
    
    Returns:
    Dictionary with validation results
    """
    validation_result = {
        'is_valid': True,
        'missing_columns': [],
        'null_counts': {},
        'data_types': {}
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_result['is_valid'] = False
            validation_result['missing_columns'] = missing
    
    for column in df.columns:
        null_count = df[column].isnull().sum()
        validation_result['null_counts'][column] = null_count
        validation_result['data_types'][column] = str(df[column].dtype)
    
    return validation_result

def example_usage():
    """Example demonstrating the data cleaning utilities."""
    sample_data = {
        'name': ['  Alice  ', 'bob', 'Alice', 'Charlie', '  BOB  '],
        'age': [25, 30, 25, 35, 30],
        'score': [85, 92, 78, 88, 95]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_column(df, 'name', remove_duplicates=True, case_sensitive=False)
    print("After cleaning 'name' column:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    normalized_df = normalize_numeric(cleaned_df, 'score', scale_range=(0, 100))
    print("After normalizing 'score' column:")
    print(normalized_df)
    print("\n" + "="*50 + "\n")
    
    validation = validate_dataframe(normalized_df, required_columns=['name', 'age', 'score'])
    print("Validation results:")
    for key, value in validation.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    example_usage()