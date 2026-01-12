import numpy as np
import pandas as pd

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method.
    Returns boolean mask where True indicates outliers.
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return (data[column] < lower_bound) | (data[column] > upper_bound)

def remove_outliers(data, columns, threshold=1.5):
    """
    Remove outliers from specified columns using IQR method.
    Returns cleaned DataFrame.
    """
    clean_data = data.copy()
    for col in columns:
        outliers = detect_outliers_iqr(clean_data, col, threshold)
        clean_data = clean_data[~outliers]
    return clean_data.reset_index(drop=True)

def normalize_minmax(data, columns):
    """
    Apply min-max normalization to specified columns.
    Returns DataFrame with normalized columns.
    """
    normalized_data = data.copy()
    for col in columns:
        min_val = normalized_data[col].min()
        max_val = normalized_data[col].max()
        normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
    return normalized_data

def normalize_zscore(data, columns):
    """
    Apply z-score normalization to specified columns.
    Returns DataFrame with normalized columns.
    """
    normalized_data = data.copy()
    for col in columns:
        mean_val = normalized_data[col].mean()
        std_val = normalized_data[col].std()
        normalized_data[col] = (normalized_data[col] - mean_val) / std_val
    return normalized_data

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy.
    Supported strategies: 'mean', 'median', 'mode', 'drop'
    """
    if columns is None:
        columns = data.columns
    
    processed_data = data.copy()
    
    if strategy == 'drop':
        return processed_data.dropna(subset=columns)
    
    for col in columns:
        if processed_data[col].isnull().any():
            if strategy == 'mean':
                fill_value = processed_data[col].mean()
            elif strategy == 'median':
                fill_value = processed_data[col].median()
            elif strategy == 'mode':
                fill_value = processed_data[col].mode()[0]
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")
            
            processed_data[col] = processed_data[col].fillna(fill_value)
    
    return processed_data

def clean_dataset(data, outlier_columns=None, normalize_columns=None, 
                  normalize_method='minmax', missing_strategy='mean'):
    """
    Comprehensive data cleaning pipeline.
    """
    cleaned_data = data.copy()
    
    if outlier_columns:
        cleaned_data = remove_outliers(cleaned_data, outlier_columns)
    
    if normalize_columns:
        if normalize_method == 'minmax':
            cleaned_data = normalize_minmax(cleaned_data, normalize_columns)
        elif normalize_method == 'zscore':
            cleaned_data = normalize_zscore(cleaned_data, normalize_columns)
    
    cleaned_data = handle_missing_values(cleaned_data, strategy=missing_strategy)
    
    return cleaned_data

def get_data_summary(data):
    """
    Generate comprehensive summary statistics for the dataset.
    """
    summary = {
        'shape': data.shape,
        'dtypes': data.dtypes.to_dict(),
        'missing_values': data.isnull().sum().to_dict(),
        'numeric_stats': data.describe().to_dict(),
        'unique_counts': {col: data[col].nunique() for col in data.columns}
    }
    return summary