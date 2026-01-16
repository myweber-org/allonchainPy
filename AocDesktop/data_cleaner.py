import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, threshold=1.5):
    """
    Detect outliers using the Interquartile Range method.
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    outliers = (data < lower_bound) | (data > upper_bound)
    return outliers, lower_bound, upper_bound

def remove_outliers_zscore(data, threshold=3):
    """
    Remove outliers using Z-score method.
    """
    z_scores = np.abs(stats.zscore(data))
    filtered_data = data[(z_scores < threshold)]
    return filtered_data

def normalize_minmax(data):
    """
    Normalize data to [0, 1] range using min-max scaling.
    """
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return np.zeros_like(data)
    normalized = (data - min_val) / (max_val - min_val)
    return normalized

def standardize_data(data):
    """
    Standardize data to have zero mean and unit variance.
    """
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return np.zeros_like(data)
    standardized = (data - mean) / std
    return standardized

def clean_dataframe(df, columns=None, outlier_method='iqr', normalize=False, standardize=False):
    """
    Clean a pandas DataFrame by handling outliers and optionally normalizing/standardizing.
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col not in df.columns:
            continue
            
        data = df[col].dropna().values
        
        if outlier_method == 'iqr':
            outliers, lower, upper = detect_outliers_iqr(data)
            df_clean.loc[outliers, col] = np.nan
        elif outlier_method == 'zscore':
            filtered_data = remove_outliers_zscore(data)
            df_clean[col] = df_clean[col].where(df_clean[col].isin(filtered_data), np.nan)
        
        if normalize:
            normalized_data = normalize_minmax(df_clean[col].dropna().values)
            df_clean.loc[df_clean[col].notna(), col] = normalized_data
        elif standardize:
            standardized_data = standardize_data(df_clean[col].dropna().values)
            df_clean.loc[df_clean[col].notna(), col] = standardized_data
    
    return df_clean

def summarize_cleaning(df_original, df_cleaned):
    """
    Generate summary statistics before and after cleaning.
    """
    summary = pd.DataFrame({
        'original_rows': df_original.shape[0],
        'cleaned_rows': df_cleaned.shape[0],
        'original_columns': df_original.shape[1],
        'cleaned_columns': df_cleaned.shape[1],
        'original_nulls': df_original.isnull().sum().sum(),
        'cleaned_nulls': df_cleaned.isnull().sum().sum(),
        'removed_rows': df_original.shape[0] - df_cleaned.shape[0]
    }, index=['summary'])
    
    return summary