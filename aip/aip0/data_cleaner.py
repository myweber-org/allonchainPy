
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, columns, factor=1.5):
    """
    Remove outliers using Interquartile Range method
    """
    df_clean = dataframe.copy()
    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def remove_outliers_zscore(dataframe, columns, threshold=3):
    """
    Remove outliers using Z-score method
    """
    df_clean = dataframe.copy()
    for col in columns:
        if col in df_clean.columns:
            z_scores = np.abs(stats.zscore(df_clean[col]))
            df_clean = df_clean[z_scores < threshold]
    return df_clean

def normalize_minmax(dataframe, columns):
    """
    Normalize data using Min-Max scaling
    """
    df_normalized = dataframe.copy()
    for col in columns:
        if col in df_normalized.columns:
            min_val = df_normalized[col].min()
            max_val = df_normalized[col].max()
            if max_val != min_val:
                df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
    return df_normalized

def normalize_zscore(dataframe, columns):
    """
    Normalize data using Z-score standardization
    """
    df_normalized = dataframe.copy()
    for col in columns:
        if col in df_normalized.columns:
            mean_val = df_normalized[col].mean()
            std_val = df_normalized[col].std()
            if std_val != 0:
                df_normalized[col] = (df_normalized[col] - mean_val) / std_val
    return df_normalized

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values with different strategies
    """
    df_filled = dataframe.copy()
    if columns is None:
        columns = df_filled.columns
    
    for col in columns:
        if col in df_filled.columns and df_filled[col].isnull().any():
            if strategy == 'mean':
                fill_value = df_filled[col].mean()
            elif strategy == 'median':
                fill_value = df_filled[col].median()
            elif strategy == 'mode':
                fill_value = df_filled[col].mode()[0]
            elif strategy == 'drop':
                df_filled = df_filled.dropna(subset=[col])
                continue
            else:
                fill_value = 0
            
            df_filled[col] = df_filled[col].fillna(fill_value)
    
    return df_filled

def clean_data_pipeline(dataframe, config):
    """
    Execute complete data cleaning pipeline based on configuration
    """
    df_clean = dataframe.copy()
    
    if 'missing_values' in config:
        df_clean = handle_missing_values(
            df_clean,
            strategy=config['missing_values'].get('strategy', 'mean'),
            columns=config['missing_values'].get('columns')
        )
    
    if 'outliers' in config:
        method = config['outliers'].get('method', 'iqr')
        columns = config['outliers'].get('columns', df_clean.columns.tolist())
        
        if method == 'iqr':
            factor = config['outliers'].get('factor', 1.5)
            df_clean = remove_outliers_iqr(df_clean, columns, factor)
        elif method == 'zscore':
            threshold = config['outliers'].get('threshold', 3)
            df_clean = remove_outliers_zscore(df_clean, columns, threshold)
    
    if 'normalization' in config:
        method = config['normalization'].get('method', 'minmax')
        columns = config['normalization'].get('columns', df_clean.columns.tolist())
        
        if method == 'minmax':
            df_clean = normalize_minmax(df_clean, columns)
        elif method == 'zscore':
            df_clean = normalize_zscore(df_clean, columns)
    
    return df_clean