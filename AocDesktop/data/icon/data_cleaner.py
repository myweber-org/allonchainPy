
import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to analyze
        threshold: IQR multiplier (default 1.5)
    
    Returns:
        Boolean mask of outliers
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    return (data[column] < lower_bound) | (data[column] > upper_bound)

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        threshold: Z-score threshold (default 3)
    
    Returns:
        DataFrame with outliers removed
    """
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    filtered_data = data[(z_scores < threshold) | (data[column].isna())]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Normalized Series
    """
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    return (data[column] - min_val) / (max_val - min_val)

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Standardized Series
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    return (data[column] - mean_val) / std_val

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='zscore'):
    """
    Main function to clean dataset by handling outliers and normalization.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process
        outlier_method: 'iqr' or 'zscore'
        normalize_method: 'minmax' or 'zscore'
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        # Handle outliers
        if outlier_method == 'iqr':
            outliers = detect_outliers_iqr(cleaned_data, column)
            cleaned_data.loc[outliers, column] = np.nan
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, column)
        
        # Normalize data
        if normalize_method == 'minmax':
            cleaned_data[f'{column}_normalized'] = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data[f'{column}_normalized'] = normalize_zscore(cleaned_data, column)
    
    return cleaned_data

def get_cleaning_summary(data, cleaned_data, numeric_columns=None):
    """
    Generate summary statistics before and after cleaning.
    
    Args:
        data: original DataFrame
        cleaned_data: cleaned DataFrame
        numeric_columns: list of numeric columns to compare
    
    Returns:
        Dictionary with summary statistics
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    summary = {}
    
    for column in numeric_columns:
        if column not in data.columns:
            continue
            
        original_stats = {
            'mean': data[column].mean(),
            'std': data[column].std(),
            'min': data[column].min(),
            'max': data[column].max(),
            'missing': data[column].isna().sum()
        }
        
        cleaned_stats = {
            'mean': cleaned_data[column].mean(),
            'std': cleaned_data[column].std(),
            'min': cleaned_data[column].min(),
            'max': cleaned_data[column].max(),
            'missing': cleaned_data[column].isna().sum()
        }
        
        summary[column] = {
            'original': original_stats,
            'cleaned': cleaned_stats,
            'outliers_removed': data[column].count() - cleaned_data[column].count()
        }
    
    return summary