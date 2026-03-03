
def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    series = data[column].dropna()
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return outliers.index.tolist()

def remove_outliers(data, column, threshold=1.5):
    """
    Remove outliers from specified column
    """
    outlier_indices = detect_outliers_iqr(data, column, threshold)
    return data.drop(index=outlier_indices)

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    series = data[column].dropna()
    min_val = series.min()
    max_val = series.max()
    
    if max_val == min_val:
        return series
    
    normalized = (series - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    series = data[column].dropna()
    mean_val = series.mean()
    std_val = series.std()
    
    if std_val == 0:
        return series
    
    standardized = (series - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy
    """
    if columns is None:
        columns = data.columns
    
    data_copy = data.copy()
    
    for column in columns:
        if column not in data_copy.columns:
            continue
            
        if data_copy[column].isnull().any():
            if strategy == 'mean':
                fill_value = data_copy[column].mean()
            elif strategy == 'median':
                fill_value = data_copy[column].median()
            elif strategy == 'mode':
                fill_value = data_copy[column].mode()[0]
            elif strategy == 'drop':
                data_copy = data_copy.dropna(subset=[column])
                continue
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            data_copy[column] = data_copy[column].fillna(fill_value)
    
    return data_copy

def clean_dataset(data, config):
    """
    Comprehensive data cleaning pipeline
    """
    cleaned_data = data.copy()
    
    # Handle missing values
    if 'missing_strategy' in config:
        cleaned_data = handle_missing_values(
            cleaned_data, 
            strategy=config['missing_strategy'],
            columns=config.get('columns')
        )
    
    # Remove outliers
    if 'remove_outliers' in config and config['remove_outliers']:
        for column in config.get('outlier_columns', []):
            if column in cleaned_data.columns:
                cleaned_data = remove_outliers(cleaned_data, column)
    
    # Normalize/Standardize
    if 'normalization' in config:
        for column, method in config['normalization'].items():
            if column in cleaned_data.columns:
                if method == 'minmax':
                    cleaned_data[column] = normalize_minmax(cleaned_data, column)
                elif method == 'zscore':
                    cleaned_data[column] = standardize_zscore(cleaned_data, column)
    
    return cleaned_data