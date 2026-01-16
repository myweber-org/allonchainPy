import pandas as pd
import numpy as np

def clean_dataset(df, deduplicate=True, fill_na=True):
    """
    Clean a pandas DataFrame by removing duplicates and handling null values.
    
    Args:
        df: pandas DataFrame to clean
        deduplicate: If True, remove duplicate rows
        fill_na: If True, fill null values with appropriate defaults
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if deduplicate:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_na:
        for column in cleaned_df.columns:
            if cleaned_df[column].dtype == 'object':
                cleaned_df[column] = cleaned_df[column].fillna('Unknown')
            elif cleaned_df[column].dtype in ['int64', 'float64']:
                cleaned_df[column] = cleaned_df[column].fillna(0)
            elif cleaned_df[column].dtype == 'bool':
                cleaned_df[column] = cleaned_df[column].fillna(False)
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate data quality by checking for required columns and data types.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of column names that must be present
    
    Returns:
        Boolean indicating if validation passed
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

def sample_data(df, n=5, random_state=42):
    """
    Return a random sample of the DataFrame for inspection.
    
    Args:
        df: pandas DataFrame to sample
        n: Number of rows to sample
        random_state: Random seed for reproducibility
    
    Returns:
        Sampled DataFrame
    """
    return df.sample(n=n, random_state=random_state)