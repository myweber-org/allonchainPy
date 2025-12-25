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
    print(f"Cleaned data saved to: {output_path}")