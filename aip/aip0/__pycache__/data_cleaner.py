
import pandas as pd
import numpy as np
from typing import Optional, Union, List

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

def handle_missing_values(df: pd.DataFrame, 
                         strategy: str = 'mean',
                         columns: Optional[List[str]] = None) -> pd.DataFrame:
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
        if df_copy[col].isnull().any():
            if strategy == 'mean':
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)
            elif strategy == 'median':
                df_copy[col].fillna(df_copy[col].median(), inplace=True)
            elif strategy == 'mode':
                df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                df_copy = df_copy.dropna(subset=[col])
    
    return df_copy

def normalize_column(df: pd.DataFrame, 
                    column: str,
                    method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize a column in DataFrame.
    
    Args:
        df: Input DataFrame
        column: Column to normalize
        method: 'minmax' or 'zscore'
    
    Returns:
        DataFrame with normalized column
    """
    df_copy = df.copy()
    
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

def detect_outliers(df: pd.DataFrame, 
                   column: str,
                   method: str = 'iqr',
                   threshold: float = 1.5) -> pd.Series:
    """
    Detect outliers in a column.
    
    Args:
        df: Input DataFrame
        column: Column to check for outliers
        method: 'iqr' or 'zscore'
        threshold: Threshold for outlier detection
    
    Returns:
        Boolean Series indicating outliers
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (df[column] < lower_bound) | (df[column] > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        return z_scores > threshold
    
    return pd.Series([False] * len(df))

def clean_dataframe(df: pd.DataFrame,
                   remove_dup: bool = True,
                   handle_na: bool = True,
                   na_strategy: str = 'mean',
                   normalize_cols: Optional[List[str]] = None,
                   outlier_method: Optional[str] = None) -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        remove_dup: Whether to remove duplicates
        handle_na: Whether to handle missing values
        na_strategy: Strategy for handling missing values
        normalize_cols: Columns to normalize
        outlier_method: Method for outlier detection
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if remove_dup:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if handle_na:
        cleaned_df = handle_missing_values(cleaned_df, strategy=na_strategy)
    
    if normalize_cols:
        for col in normalize_cols:
            if col in cleaned_df.columns:
                cleaned_df = normalize_column(cleaned_df, col)
    
    if outlier_method:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            outliers = detect_outliers(cleaned_df, col, method=outlier_method)
            if outliers.any():
                cleaned_df = cleaned_df[~outliers]
    
    return cleaned_df