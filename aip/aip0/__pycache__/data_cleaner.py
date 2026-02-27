
import pandas as pd
import numpy as np

def remove_missing_rows(df, columns=None):
    """
    Remove rows with missing values from specified columns or entire DataFrame.
    
    Args:
        df: pandas DataFrame
        columns: list of column names or None for all columns
    
    Returns:
        Cleaned DataFrame
    """
    if columns:
        return df.dropna(subset=columns)
    return df.dropna()

def fill_missing_with_mean(df, columns):
    """
    Fill missing values in specified columns with column mean.
    
    Args:
        df: pandas DataFrame
        columns: list of column names
    
    Returns:
        DataFrame with filled values
    """
    df_filled = df.copy()
    for col in columns:
        if col in df.columns:
            df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
    return df_filled

def detect_outliers_iqr(df, column, multiplier=1.5):
    """
    Detect outliers using Interquartile Range method.
    
    Args:
        df: pandas DataFrame
        column: column name to check for outliers
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        Boolean Series indicating outliers
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def cap_outliers(df, column, method='iqr', multiplier=1.5):
    """
    Cap outliers to specified bounds.
    
    Args:
        df: pandas DataFrame
        column: column name to process
        method: 'iqr' or 'percentile'
        multiplier: IQR multiplier (for 'iqr' method)
    
    Returns:
        DataFrame with capped values
    """
    df_capped = df.copy()
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
    elif method == 'percentile':
        lower_bound = df[column].quantile(0.01)
        upper_bound = df[column].quantile(0.99)
    
    df_capped[column] = df_capped[column].clip(lower=lower_bound, upper=upper_bound)
    return df_capped

def standardize_columns(df, columns):
    """
    Standardize specified columns to have zero mean and unit variance.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to standardize
    
    Returns:
        DataFrame with standardized columns
    """
    df_standardized = df.copy()
    for col in columns:
        if col in df.columns:
            mean = df_standardized[col].mean()
            std = df_standardized[col].std()
            if std > 0:
                df_standardized[col] = (df_standardized[col] - mean) / std
    return df_standardized

def clean_dataset(df, missing_strategy='remove', outlier_strategy='cap', 
                  columns_to_clean=None, standardize_cols=None):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: pandas DataFrame
        missing_strategy: 'remove' or 'mean'
        outlier_strategy: 'cap' or 'ignore'
        columns_to_clean: list of columns to process
        standardize_cols: list of columns to standardize
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if columns_to_clean is None:
        columns_to_clean = df.columns.tolist()
    
    numeric_cols = df[columns_to_clean].select_dtypes(include=[np.number]).columns.tolist()
    
    if missing_strategy == 'remove':
        cleaned_df = remove_missing_rows(cleaned_df, columns=numeric_cols)
    elif missing_strategy == 'mean':
        cleaned_df = fill_missing_with_mean(cleaned_df, columns=numeric_cols)
    
    if outlier_strategy == 'cap':
        for col in numeric_cols:
            if detect_outliers_iqr(cleaned_df, col).any():
                cleaned_df = cap_outliers(cleaned_df, col)
    
    if standardize_cols:
        cleaned_df = standardize_columns(cleaned_df, standardize_cols)
    
    return cleaned_df