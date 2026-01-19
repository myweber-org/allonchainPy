
import pandas as pd
import numpy as np
from typing import Union, List, Dict, Optional

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

def normalize_data(df: pd.DataFrame,
                  columns: Optional[List[str]] = None,
                  method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize numerical columns in DataFrame.
    
    Args:
        df: Input DataFrame
        columns: Columns to normalize
        method: 'minmax' or 'zscore'
    
    Returns:
        Normalized DataFrame
    """
    df_copy = df.copy()
    
    if columns is None:
        columns = df_copy.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if method == 'minmax':
            min_val = df_copy[col].min()
            max_val = df_copy[col].max()
            if max_val > min_val:
                df_copy[col] = (df_copy[col] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = df_copy[col].mean()
            std_val = df_copy[col].std()
            if std_val > 0:
                df_copy[col] = (df_copy[col] - mean_val) / std_val
    
    return df_copy

def detect_outliers(df: pd.DataFrame,
                   columns: Optional[List[str]] = None,
                   method: str = 'iqr',
                   threshold: float = 1.5) -> Dict[str, List[int]]:
    """
    Detect outliers in numerical columns.
    
    Args:
        df: Input DataFrame
        columns: Columns to check for outliers
        method: 'iqr' or 'zscore'
        threshold: Threshold for outlier detection
    
    Returns:
        Dictionary with column names as keys and outlier indices as values
    """
    outliers = {}
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        col_data = df[col].dropna()
        
        if method == 'iqr':
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
        
        elif method == 'zscore':
            mean_val = col_data.mean()
            std_val = col_data.std()
            if std_val > 0:
                z_scores = np.abs((df[col] - mean_val) / std_val)
                outlier_indices = df[z_scores > threshold].index.tolist()
            else:
                outlier_indices = []
        
        if outlier_indices:
            outliers[col] = outlier_indices
    
    return outliers

def clean_dataset(df: pd.DataFrame,
                 remove_dup: bool = True,
                 handle_na: bool = True,
                 na_strategy: str = 'mean',
                 normalize: bool = False,
                 norm_method: str = 'minmax') -> pd.DataFrame:
    """
    Comprehensive dataset cleaning pipeline.
    
    Args:
        df: Input DataFrame
        remove_dup: Whether to remove duplicates
        handle_na: Whether to handle missing values
        na_strategy: Strategy for handling missing values
        normalize: Whether to normalize data
        norm_method: Normalization method
    
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if remove_dup:
        df_clean = remove_duplicates(df_clean)
    
    if handle_na:
        df_clean = handle_missing_values(df_clean, strategy=na_strategy)
    
    if normalize:
        df_clean = normalize_data(df_clean, method=norm_method)
    
    return df_clean