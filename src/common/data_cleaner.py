
import pandas as pd
import numpy as np
from typing import Union, List, Dict

def remove_duplicates(df: pd.DataFrame, subset: Union[List[str], None] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean', columns: Union[List[str], None] = None) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: 'mean', 'median', 'mode', or 'drop'
        columns: Specific columns to process
    
    Returns:
        DataFrame with handled missing values
    """
    df_copy = df.copy()
    
    if columns is None:
        columns = df_copy.columns
    
    for col in columns:
        if df_copy[col].dtype in ['int64', 'float64']:
            if strategy == 'mean':
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)
            elif strategy == 'median':
                df_copy[col].fillna(df_copy[col].median(), inplace=True)
            elif strategy == 'mode':
                df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                df_copy = df_copy.dropna(subset=[col])
        else:
            if strategy == 'mode':
                df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                df_copy = df_copy.dropna(subset=[col])
    
    return df_copy

def normalize_columns(df: pd.DataFrame, columns: Union[List[str], None] = None) -> pd.DataFrame:
    """
    Normalize numerical columns to range [0, 1].
    
    Args:
        df: Input DataFrame
        columns: Specific columns to normalize
    
    Returns:
        DataFrame with normalized columns
    """
    df_copy = df.copy()
    
    if columns is None:
        columns = [col for col in df_copy.columns if df_copy[col].dtype in ['int64', 'float64']]
    
    for col in columns:
        if df_copy[col].dtype in ['int64', 'float64']:
            col_min = df_copy[col].min()
            col_max = df_copy[col].max()
            if col_max != col_min:
                df_copy[col] = (df_copy[col] - col_min) / (col_max - col_min)
    
    return df_copy

def detect_outliers_iqr(df: pd.DataFrame, columns: Union[List[str], None] = None) -> Dict[str, List[int]]:
    """
    Detect outliers using IQR method.
    
    Args:
        df: Input DataFrame
        columns: Specific columns to check
    
    Returns:
        Dictionary with column names as keys and outlier indices as values
    """
    if columns is None:
        columns = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
    
    outliers = {}
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
        if outlier_indices:
            outliers[col] = outlier_indices
    
    return outliers

def clean_dataset(df: pd.DataFrame, 
                  remove_dups: bool = True,
                  handle_nulls: Union[str, None] = 'mean',
                  normalize: bool = False) -> pd.DataFrame:
    """
    Comprehensive dataset cleaning pipeline.
    
    Args:
        df: Input DataFrame
        remove_dups: Whether to remove duplicates
        handle_nulls: Strategy for handling nulls or None to skip
        normalize: Whether to normalize numerical columns
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if remove_dups:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if handle_nulls:
        cleaned_df = handle_missing_values(cleaned_df, strategy=handle_nulls)
    
    if normalize:
        cleaned_df = normalize_columns(cleaned_df)
    
    return cleaned_df