import pandas as pd
import numpy as np

def remove_missing_rows(df, columns=None):
    """
    Remove rows with missing values from DataFrame.
    
    Args:
        df: pandas DataFrame
        columns: list of columns to check for missing values.
                 If None, checks all columns.
    
    Returns:
        Cleaned DataFrame
    """
    if columns is None:
        columns = df.columns
    
    return df.dropna(subset=columns)

def fill_missing_with_mean(df, columns=None):
    """
    Fill missing values with column mean.
    
    Args:
        df: pandas DataFrame
        columns: list of columns to fill.
                 If None, fills all numeric columns.
    
    Returns:
        DataFrame with filled values
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_filled = df.copy()
    for col in columns:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            df_filled[col] = df[col].fillna(df[col].mean())
    
    return df_filled

def detect_outliers_iqr(df, column, multiplier=1.5):
    """
    Detect outliers using IQR method.
    
    Args:
        df: pandas DataFrame
        column: column name to check for outliers
        multiplier: IQR multiplier (default: 1.5)
    
    Returns:
        Boolean Series indicating outliers
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def remove_outliers(df, column, multiplier=1.5):
    """
    Remove outliers from DataFrame.
    
    Args:
        df: pandas DataFrame
        column: column name to check for outliers
        multiplier: IQR multiplier (default: 1.5)
    
    Returns:
        DataFrame without outliers
    """
    outliers = detect_outliers_iqr(df, column, multiplier)
    return df[~outliers]

def normalize_column(df, column, method='minmax'):
    """
    Normalize column values.
    
    Args:
        df: pandas DataFrame
        column: column name to normalize
        method: normalization method ('minmax' or 'zscore')
    
    Returns:
        DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_normalized = df.copy()
    
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val != min_val:
            df_normalized[column] = (df[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val != 0:
            df_normalized[column] = (df[column] - mean_val) / std_val
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df_normalized

def clean_dataset(df, numeric_columns=None, outlier_multiplier=1.5):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: pandas DataFrame
        numeric_columns: list of numeric columns to process
        outlier_multiplier: IQR multiplier for outlier detection
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove rows with missing values in numeric columns
    df_clean = remove_missing_rows(df, numeric_columns)
    
    # Fill remaining missing values with mean
    df_clean = fill_missing_with_mean(df_clean, numeric_columns)
    
    # Remove outliers from each numeric column
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean = remove_outliers(df_clean, col, outlier_multiplier)
    
    # Normalize numeric columns
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean = normalize_column(df_clean, col, method='minmax')
    
    return df_clean.reset_index(drop=True)