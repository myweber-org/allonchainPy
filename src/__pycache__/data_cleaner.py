
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns=None, threshold=1.5):
    """
    Remove outliers using IQR method.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean

def remove_outliers_zscore(df, columns=None, threshold=3):
    """
    Remove outliers using Z-score method.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    for col in columns:
        z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
        df_clean = df_clean[(z_scores < threshold) | (df_clean[col].isna())]
    
    return df_clean

def normalize_minmax(df, columns=None):
    """
    Normalize data using Min-Max scaling.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    for col in columns:
        min_val = df_normalized[col].min()
        max_val = df_normalized[col].max()
        if max_val != min_val:
            df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
    
    return df_normalized

def normalize_zscore(df, columns=None):
    """
    Normalize data using Z-score standardization.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    for col in columns:
        mean_val = df_normalized[col].mean()
        std_val = df_normalized[col].std()
        if std_val != 0:
            df_normalized[col] = (df_normalized[col] - mean_val) / std_val
    
    return df_normalized

def clean_dataset(df, outlier_method='iqr', normalize_method='minmax', outlier_threshold=1.5, normalize_columns=None):
    """
    Complete data cleaning pipeline.
    """
    if outlier_method == 'iqr':
        df_clean = remove_outliers_iqr(df, threshold=outlier_threshold)
    elif outlier_method == 'zscore':
        df_clean = remove_outliers_zscore(df, threshold=outlier_threshold)
    else:
        df_clean = df.copy()
    
    if normalize_method == 'minmax':
        df_final = normalize_minmax(df_clean, columns=normalize_columns)
    elif normalize_method == 'zscore':
        df_final = normalize_zscore(df_clean, columns=normalize_columns)
    else:
        df_final = df_clean
    
    return df_final

def validate_data(df, check_missing=True, check_inf=True):
    """
    Validate data quality.
    """
    validation_report = {}
    
    if check_missing:
        missing_counts = df.isnull().sum()
        validation_report['missing_values'] = missing_counts[missing_counts > 0].to_dict()
    
    if check_inf:
        inf_counts = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_counts[col] = inf_count
        validation_report['infinite_values'] = inf_counts
    
    validation_report['total_rows'] = len(df)
    validation_report['total_columns'] = len(df.columns)
    validation_report['data_types'] = df.dtypes.to_dict()
    
    return validation_report