import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df, strategy='mean', columns=None):
    """
    Fill missing values in DataFrame.
    
    Args:
        df: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'constant'
        columns: list of columns to fill, None for all columns
    
    Returns:
        DataFrame with missing values filled
    """
    df_filled = df.copy()
    
    if columns is None:
        columns = df.columns
    
    for col in columns:
        if df[col].dtype in ['int64', 'float64']:
            if strategy == 'mean':
                df_filled[col] = df[col].fillna(df[col].mean())
            elif strategy == 'median':
                df_filled[col] = df[col].fillna(df[col].median())
            elif strategy == 'mode':
                df_filled[col] = df[col].fillna(df[col].mode()[0])
            elif strategy == 'constant':
                df_filled[col] = df[col].fillna(0)
        else:
            df_filled[col] = df[col].fillna(df[col].mode()[0])
    
    return df_filled

def remove_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Remove outliers from DataFrame using specified method.
    
    Args:
        df: pandas DataFrame
        columns: list of numeric columns to check for outliers
        method: 'iqr' for interquartile range, 'zscore' for standard deviation
        threshold: multiplier for IQR or cutoff for z-score
    
    Returns:
        DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    
    if method == 'iqr':
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    elif method == 'zscore':
        for col in columns:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df_clean = df_clean[z_scores < threshold]
    
    return df_clean.reset_index(drop=True)

def standardize_columns(df, columns=None):
    """
    Standardize numeric columns to have zero mean and unit variance.
    
    Args:
        df: pandas DataFrame
        columns: list of columns to standardize
    
    Returns:
        DataFrame with standardized columns
    """
    df_std = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            df_std[col] = (df[col] - mean) / std
    
    return df_std

def clean_dataset(df, remove_dup=True, fill_na=True, remove_out=True, standardize=True):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: pandas DataFrame
        remove_dup: whether to remove duplicates
        fill_na: whether to fill missing values
        remove_out: whether to remove outliers
        standardize: whether to standardize numeric columns
    
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if remove_dup:
        df_clean = remove_duplicates(df_clean)
    
    if fill_na:
        df_clean = fill_missing_values(df_clean)
    
    if remove_out:
        df_clean = remove_outliers(df_clean)
    
    if standardize:
        df_clean = standardize_columns(df_clean)
    
    return df_clean