import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_na=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows.
        fill_na (bool): Whether to fill missing values.
        fill_value: Value to use for filling missing data.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()
    
    if fill_na:
        df_clean = df_clean.fillna(fill_value)
    
    return df_clean

def remove_outliers(df, column, threshold=3):
    """
    Remove outliers from a specific column using z-score method.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to process.
        threshold (float): Z-score threshold for outlier detection.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    return df[z_scores < threshold]

def normalize_column(df, column):
    """
    Normalize a column to range [0, 1] using min-max scaling.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to normalize.
    
    Returns:
        pd.DataFrame: DataFrame with normalized column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_normalized = df.copy()
    min_val = df[column].min()
    max_val = df[column].max()
    
    if max_val > min_val:
        df_normalized[column] = (df[column] - min_val) / (max_val - min_val)
    
    return df_normalized