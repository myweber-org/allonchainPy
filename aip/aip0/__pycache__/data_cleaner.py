
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
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

def clean_dataset(df, numeric_columns=None, outlier_method='iqr', normalize_method='zscore'):
    """
    Comprehensive data cleaning pipeline
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    report = {
        'original_rows': len(df),
        'outliers_removed': {},
        'columns_normalized': []
    }
    
    for col in numeric_columns:
        if col not in df.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_df, removed = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df, removed = remove_outliers_zscore(cleaned_df, col)
        else:
            raise ValueError(f"Unknown outlier method: {outlier_method}")
        
        report['outliers_removed'][col] = removed
        
        if normalize_method == 'minmax':
            cleaned_df[f'{col}_normalized'] = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df[f'{col}_normalized'] = normalize_zscore(cleaned_df, col)
        else:
            raise ValueError(f"Unknown normalize method: {normalize_method}")
        
        report['columns_normalized'].append(col)
    
    report['final_rows'] = len(cleaned_df)
    report['rows_removed'] = report['original_rows'] - report['final_rows']
    
    return cleaned_df, report

def validate_data(df, required_columns=None, allow_nan_ratio=0.1):
    """
    Validate dataset structure and completeness
    """
    validation_report = {
        'is_valid': True,
        'missing_columns': [],
        'high_nan_columns': [],
        'validation_errors': []
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_report['missing_columns'] = missing
            validation_report['is_valid'] = False
            validation_report['validation_errors'].append(f"Missing required columns: {missing}")
    
    for col in df.columns:
        nan_ratio = df[col].isna().sum() / len(df)
        if nan_ratio > allow_nan_ratio:
            validation_report['high_nan_columns'].append({
                'column': col,
                'nan_ratio': nan_ratio
            })
            validation_report['is_valid'] = False
    
    return validation_report