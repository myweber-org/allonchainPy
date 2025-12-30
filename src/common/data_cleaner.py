
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
        if col in df_copy.columns:
            if strategy == 'drop':
                df_copy = df_copy.dropna(subset=[col])
            elif strategy == 'mean':
                df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
            elif strategy == 'median':
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())
            elif strategy == 'mode':
                df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])
    
    return df_copy

def normalize_columns(df: pd.DataFrame, columns: Union[List[str], None] = None) -> pd.DataFrame:
    """
    Normalize specified columns to range [0, 1].
    
    Args:
        df: Input DataFrame
        columns: Columns to normalize
    
    Returns:
        DataFrame with normalized columns
    """
    df_copy = df.copy()
    
    if columns is None:
        columns = df_copy.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
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
        columns: Columns to check for outliers
    
    Returns:
        Dictionary with column names as keys and outlier indices as values
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    outliers = {}
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
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
                  missing_strategy: str = 'mean',
                  normalize: bool = False,
                  outlier_threshold: float = 0.0) -> pd.DataFrame:
    """
    Comprehensive dataset cleaning pipeline.
    
    Args:
        df: Input DataFrame
        remove_dups: Whether to remove duplicates
        missing_strategy: Strategy for handling missing values
        normalize: Whether to normalize numeric columns
        outlier_threshold: Threshold for outlier removal (0.0 = no removal)
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if remove_dups:
        cleaned_df = remove_duplicates(cleaned_df)
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    if normalize:
        cleaned_df = normalize_columns(cleaned_df)
    
    if outlier_threshold > 0:
        outliers = detect_outliers_iqr(cleaned_df)
        for col, indices in outliers.items():
            if len(indices) / len(cleaned_df) <= outlier_threshold:
                cleaned_df = cleaned_df.drop(indices)
    
    return cleaned_df
import pandas as pd
import re

def clean_dataset(df, column_names):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing specified string columns.
    """
    # Remove duplicate rows
    df_clean = df.drop_duplicates().reset_index(drop=True)
    
    # Normalize string columns: trim whitespace and convert to lowercase
    for col in column_names:
        if col in df_clean.columns and df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype(str).apply(lambda x: re.sub(r'\s+', ' ', x.strip().lower()))
    
    return df_clean

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column and return a boolean mask.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    # Simple email regex pattern
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return df[email_column].astype(str).str.match(email_pattern)