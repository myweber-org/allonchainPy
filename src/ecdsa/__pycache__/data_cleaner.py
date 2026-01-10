
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.copy()

def detect_outliers_iqr(df, column):
    """
    Detect outliers in a DataFrame column using IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    pd.Series: Boolean series indicating outliers (True = outlier)
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    return outliers

def calculate_outlier_statistics(df, column):
    """
    Calculate outlier statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing outlier statistics
    """
    outliers = detect_outliers_iqr(df, column)
    
    stats = {
        'total_count': len(df),
        'outlier_count': outliers.sum(),
        'outlier_percentage': (outliers.sum() / len(df)) * 100,
        'min_value': df[column].min(),
        'max_value': df[column].max(),
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std_dev': df[column].std()
    }
    
    return stats

def clean_multiple_columns(df, columns):
    """
    Remove outliers from multiple columns sequentially.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed from all specified columns
    """
    cleaned_df = df.copy()
    
    for column in columns:
        if column in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
    
    return cleaned_df