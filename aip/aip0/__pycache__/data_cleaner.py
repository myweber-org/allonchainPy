
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        factor: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

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
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    return (data[column] - min_val) / (max_val - min_val)

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    return (data[column] - mean_val) / std_val

def clean_dataset(data, numeric_columns=None, method='iqr', norm_method='minmax'):
    """
    Comprehensive data cleaning function.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to clean (default: all numeric)
        method: outlier removal method ('iqr' or 'zscore')
        norm_method: normalization method ('minmax' or 'zscore')
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        if method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
        elif method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, column)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        if norm_method == 'minmax':
            cleaned_data[f'{column}_normalized'] = normalize_minmax(cleaned_data, column)
        elif norm_method == 'zscore':
            cleaned_data[f'{column}_normalized'] = normalize_zscore(cleaned_data, column)
        else:
            raise ValueError(f"Unknown normalization method: {norm_method}")
    
    return cleaned_data

def get_summary_statistics(data):
    """
    Get summary statistics for numeric columns.
    
    Args:
        data: pandas DataFrame
    
    Returns:
        DataFrame with summary statistics
    """
    numeric_data = data.select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        return pd.DataFrame()
    
    summary = numeric_data.describe().T
    summary['skewness'] = numeric_data.skew()
    summary['kurtosis'] = numeric_data.kurtosis()
    summary['missing'] = numeric_data.isnull().sum()
    summary['missing_pct'] = (numeric_data.isnull().sum() / len(data)) * 100
    
    return summary

def detect_skewed_columns(data, threshold=0.5):
    """
    Detect columns with significant skewness.
    
    Args:
        data: pandas DataFrame
        threshold: absolute skewness threshold (default 0.5)
    
    Returns:
        List of skewed column names
    """
    numeric_data = data.select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        return []
    
    skewness = numeric_data.skew()
    skewed_columns = skewness[abs(skewness) > threshold].index.tolist()
    
    return skewed_columns