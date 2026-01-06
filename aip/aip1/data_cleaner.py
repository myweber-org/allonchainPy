
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    outliers_removed = len(data) - len(filtered_data)
    
    return filtered_data, outliers_removed

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    outliers_removed = len(data) - len(filtered_data)
    
    return filtered_data, outliers_removed

def normalize_minmax(data, columns=None):
    """
    Normalize data using Min-Max scaling.
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    normalized_data = data.copy()
    for col in columns:
        if col in data.columns:
            min_val = data[col].min()
            max_val = data[col].max()
            if max_val != min_val:
                normalized_data[col] = (data[col] - min_val) / (max_val - min_val)
    
    return normalized_data

def normalize_zscore(data, columns=None):
    """
    Normalize data using Z-score standardization.
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    standardized_data = data.copy()
    for col in columns:
        if col in data.columns:
            mean_val = data[col].mean()
            std_val = data[col].std()
            if std_val > 0:
                standardized_data[col] = (data[col] - mean_val) / std_val
    
    return standardized_data

def clean_dataset(data, outlier_method='iqr', outlier_columns=None, 
                  normalization_method=None, norm_columns=None):
    """
    Main function to clean dataset with optional outlier removal and normalization.
    """
    cleaned_data = data.copy()
    report = {}
    
    if outlier_method and outlier_columns:
        total_outliers = 0
        for col in outlier_columns:
            if col in cleaned_data.columns:
                if outlier_method == 'iqr':
                    cleaned_data, outliers = remove_outliers_iqr(cleaned_data, col)
                elif outlier_method == 'zscore':
                    cleaned_data, outliers = remove_outliers_zscore(cleaned_data, col)
                else:
                    raise ValueError(f"Unknown outlier method: {outlier_method}")
                total_outliers += outliers
                report[f'outliers_removed_{col}'] = outliers
        
        report['total_outliers_removed'] = total_outliers
    
    if normalization_method and norm_columns:
        if normalization_method == 'minmax':
            cleaned_data = normalize_minmax(cleaned_data, norm_columns)
        elif normalization_method == 'zscore':
            cleaned_data = normalize_zscore(cleaned_data, norm_columns)
        else:
            raise ValueError(f"Unknown normalization method: {normalization_method}")
        
        report['normalization_applied'] = normalization_method
    
    return cleaned_data, report

def validate_data(data, required_columns=None, check_missing=True, 
                  check_duplicates=True, check_types=True):
    """
    Validate data for common issues.
    """
    validation_report = {}
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        validation_report['missing_columns'] = missing_columns
    
    if check_missing:
        missing_values = data.isnull().sum().sum()
        validation_report['total_missing_values'] = missing_values
    
    if check_duplicates:
        duplicate_rows = data.duplicated().sum()
        validation_report['duplicate_rows'] = duplicate_rows
    
    if check_types:
        column_types = data.dtypes.to_dict()
        validation_report['column_types'] = column_types
    
    return validation_report