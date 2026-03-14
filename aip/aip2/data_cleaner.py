
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using Interquartile Range method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline
    """
    cleaned_data = data.copy()
    removal_stats = {}
    
    for col in numeric_columns:
        if col not in cleaned_data.columns:
            continue
            
        # Remove outliers
        if outlier_method == 'iqr':
            cleaned_data, removed = remove_outliers_iqr(cleaned_data, col)
        elif outlier_method == 'zscore':
            cleaned_data, removed = remove_outliers_zscore(cleaned_data, col)
        else:
            removed = 0
        
        removal_stats[col] = removed
        
        # Normalize data
        if normalize_method == 'minmax':
            cleaned_data[col] = normalize_minmax(cleaned_data, col)
        elif normalize_method == 'zscore':
            cleaned_data[col] = normalize_zscore(cleaned_data, col)
    
    return cleaned_data, removal_stats

def validate_data(data, required_columns, numeric_threshold=0.8):
    """
    Validate dataset structure and quality
    """
    validation_results = {
        'missing_columns': [],
        'low_numeric_ratio': [],
        'validation_passed': True
    }
    
    # Check required columns
    for col in required_columns:
        if col not in data.columns:
            validation_results['missing_columns'].append(col)
            validation_results['validation_passed'] = False
    
    # Check numeric data ratio
    for col in data.select_dtypes(include=[np.number]).columns:
        non_null_count = data[col].count()
        total_count = len(data)
        
        if total_count > 0 and (non_null_count / total_count) < numeric_threshold:
            validation_results['low_numeric_ratio'].append(col)
            validation_results['validation_passed'] = False
    
    return validation_results
import pandas as pd
import numpy as np

def remove_missing_values(df, threshold=0.5):
    """
    Remove columns with missing values exceeding threshold percentage.
    """
    missing_percent = df.isnull().sum() / len(df)
    columns_to_drop = missing_percent[missing_percent > threshold].index
    df_cleaned = df.drop(columns=columns_to_drop)
    return df_cleaned

def fill_missing_with_median(df, columns=None):
    """
    Fill missing values with column median.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    df_filled = df.copy()
    for col in columns:
        if col in df.columns:
            df_filled[col] = df_filled[col].fillna(df_filled[col].median())
    return df_filled

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers using IQR method.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_filtered

def standardize_columns(df, columns=None):
    """
    Standardize numeric columns to zero mean and unit variance.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    df_standardized = df.copy()
    for col in columns:
        if col in df.columns:
            mean = df_standardized[col].mean()
            std = df_standardized[col].std()
            if std > 0:
                df_standardized[col] = (df_standardized[col] - mean) / std
    return df_standardized

def clean_dataset(df, missing_threshold=0.5, outlier_columns=None, standardize=True):
    """
    Perform comprehensive data cleaning pipeline.
    """
    df_cleaned = remove_missing_values(df, threshold=missing_threshold)
    df_cleaned = fill_missing_with_median(df_cleaned)
    
    if outlier_columns:
        for col in outlier_columns:
            if col in df_cleaned.columns:
                df_cleaned = remove_outliers_iqr(df_cleaned, col)
    
    if standardize:
        df_cleaned = standardize_columns(df_cleaned)
    
    return df_cleaned