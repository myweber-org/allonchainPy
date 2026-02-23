
import pandas as pd
import numpy as np
from typing import List, Optional

def remove_duplicate_rows(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def normalize_string_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Normalize string columns by stripping whitespace and converting to lowercase.
    
    Args:
        df: Input DataFrame
        columns: List of column names to normalize
    
    Returns:
        DataFrame with normalized string columns
    """
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns and df_copy[col].dtype == 'object':
            df_copy[col] = df_copy[col].astype(str).str.strip().str.lower()
    return df_copy

def clean_numeric_outliers(df: pd.DataFrame, columns: List[str], 
                          method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """
    Clean numeric outliers using specified method.
    
    Args:
        df: Input DataFrame
        columns: Numeric columns to clean
        method: 'iqr' for interquartile range or 'zscore' for standard deviation
        threshold: Threshold for outlier detection
    
    Returns:
        DataFrame with outliers replaced by NaN
    """
    df_copy = df.copy()
    
    for col in columns:
        if col not in df_copy.columns or not np.issubdtype(df_copy[col].dtype, np.number):
            continue
            
        if method == 'iqr':
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask = (df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((df_copy[col] - df_copy[col].mean()) / df_copy[col].std())
            mask = z_scores > threshold
            
        else:
            continue
            
        df_copy.loc[mask, col] = np.nan
    
    return df_copy

def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop', 
                         fill_value: Optional[float] = None) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: 'drop' to remove rows, 'fill' to fill values
        fill_value: Value to use when filling (only for numeric columns if None)
    
    Returns:
        DataFrame with handled missing values
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill':
        if fill_value is not None:
            return df.fillna(fill_value)
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df_copy = df.copy()
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(df_copy[numeric_cols].mean())
            return df_copy
    return df

def clean_dataframe(df: pd.DataFrame, 
                   duplicate_subset: Optional[List[str]] = None,
                   string_columns: Optional[List[str]] = None,
                   numeric_columns: Optional[List[str]] = None,
                   missing_strategy: str = 'drop') -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        duplicate_subset: Columns for duplicate removal
        string_columns: String columns to normalize
        numeric_columns: Numeric columns for outlier cleaning
        missing_strategy: Strategy for handling missing values
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    cleaned_df = remove_duplicate_rows(cleaned_df, duplicate_subset)
    
    if string_columns:
        cleaned_df = normalize_string_columns(cleaned_df, string_columns)
    
    if numeric_columns:
        cleaned_df = clean_numeric_outliers(cleaned_df, numeric_columns)
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    return cleaned_df