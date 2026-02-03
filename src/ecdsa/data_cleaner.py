import pandas as pd
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
    df_converted = df.copy()
    for column, dtype in column_types.items():
        if column in df_converted.columns:
            try:
                df_converted[column] = df_converted[column].astype(dtype)
            except (ValueError, TypeError):
                df_converted[column] = pd.to_numeric(df_converted[column], errors='coerce')
    return df_converted

def handle_missing_values(df: pd.DataFrame, 
                          strategy: str = 'mean',
                          columns: List[str] = None) -> pd.DataFrame:
    """
    Handle missing values in DataFrame columns.
    
    Args:
        df: Input DataFrame
        strategy: Imputation strategy ('mean', 'median', 'mode', 'drop')
        columns: Specific columns to process
    
    Returns:
        DataFrame with handled missing values
    """
    df_processed = df.copy()
    
    if columns is None:
        columns = df_processed.columns
    
    for column in columns:
        if column not in df_processed.columns:
            continue
            
        if strategy == 'drop':
            df_processed = df_processed.dropna(subset=[column])
        elif strategy == 'mean':
            df_processed[column] = df_processed[column].fillna(df_processed[column].mean())
        elif strategy == 'median':
            df_processed[column] = df_processed[column].fillna(df_processed[column].median())
        elif strategy == 'mode':
            df_processed[column] = df_processed[column].fillna(df_processed[column].mode()[0])
    
    return df_processed

def clean_dataframe(df: pd.DataFrame,
                    deduplicate: bool = True,
                    type_conversions: dict = None,
                    missing_strategy: str = 'mean') -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        deduplicate: Whether to remove duplicates
        type_conversions: Dictionary for column type conversions
        missing_strategy: Strategy for handling missing values
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if deduplicate:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if type_conversions:
        cleaned_df = convert_column_types(cleaned_df, type_conversions)
    
    if missing_strategy:
        cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    return cleaned_df

def validate_dataframe(df: pd.DataFrame,
                       required_columns: List[str] = None) -> bool:
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of columns that must be present
    
    Returns:
        Boolean indicating if DataFrame is valid
    """
    if df.empty:
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False
    
    return True
import pandas as pd

def clean_dataframe(df, drop_na=True, rename_columns=None):
    """
    Clean a pandas DataFrame by removing null values and standardizing column names.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_na (bool): If True, drop rows with any null values.
    rename_columns (dict): Optional dictionary mapping old column names to new names.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_na:
        cleaned_df = cleaned_df.dropna()
    
    if rename_columns:
        cleaned_df = cleaned_df.rename(columns=rename_columns)
    
    cleaned_df.columns = cleaned_df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for required columns and data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    if df.empty:
        print("DataFrame is empty")
        return False
    
    return True

def sample_data_cleaning():
    """
    Example usage of the data cleaning functions.
    """
    data = {
        'Name': ['Alice', 'Bob', None, 'David'],
        'Age': [25, 30, 35, None],
        'City': ['NYC', 'LA', 'Chicago', 'Boston']
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataframe(df, drop_na=True)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_dataframe(cleaned, required_columns=['name', 'age'])
    print(f"\nDataFrame validation: {is_valid}")

if __name__ == "__main__":
    sample_data_cleaning()