
import pandas as pd
import numpy as np
from scipy import stats

def normalize_data(df, columns=None, method='zscore'):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    
    if method == 'zscore':
        for col in columns:
            if col in df.columns:
                df_normalized[col] = stats.zscore(df[col])
    elif method == 'minmax':
        for col in columns:
            if col in df.columns:
                col_min = df[col].min()
                col_max = df[col].max()
                if col_max != col_min:
                    df_normalized[col] = (df[col] - col_min) / (col_max - col_min)
    elif method == 'robust':
        for col in columns:
            if col in df.columns:
                median = df[col].median()
                iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
                if iqr > 0:
                    df_normalized[col] = (df[col] - median) / iqr
    
    return df_normalized

def remove_outliers(df, columns=None, method='iqr', threshold=1.5):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    
    if method == 'iqr':
        for col in columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
                df_clean = df_clean[mask]
    
    elif method == 'zscore':
        for col in columns:
            if col in df.columns:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                mask = z_scores < threshold
                df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def handle_missing_values(df, columns=None, strategy='mean'):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_filled = df.copy()
    
    for col in columns:
        if col in df.columns and df[col].isnull().any():
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0]
            elif strategy == 'ffill':
                df_filled[col] = df[col].ffill()
                continue
            elif strategy == 'bfill':
                df_filled[col] = df[col].bfill()
                continue
            else:
                fill_value = 0
            
            df_filled[col] = df[col].fillna(fill_value)
    
    return df_filled

def clean_dataset(df, numeric_columns=None, 
                  normalize_method=None,
                  outlier_method=None,
                  outlier_threshold=1.5,
                  missing_strategy='mean'):
    
    df_cleaned = df.copy()
    
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_cleaned = handle_missing_values(df_cleaned, numeric_columns, missing_strategy)
    
    if outlier_method:
        df_cleaned = remove_outliers(df_cleaned, numeric_columns, 
                                    outlier_method, outlier_threshold)
    
    if normalize_method:
        df_cleaned = normalize_data(df_cleaned, numeric_columns, normalize_method)
    
    return df_cleaned