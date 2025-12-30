
import pandas as pd
import numpy as np
from scipy import stats

def normalize_data(df, columns=None, method='zscore'):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            if method == 'zscore':
                df_normalized[col] = (df[col] - df[col].mean()) / df[col].std()
            elif method == 'minmax':
                df_normalized[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            elif method == 'robust':
                median = df[col].median()
                iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
                df_normalized[col] = (df[col] - median) / iqr if iqr != 0 else 0
    
    return df_normalized

def remove_outliers(df, columns=None, method='iqr', threshold=1.5):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                mask = z_scores < threshold
                mask = mask.reindex(df.index, fill_value=True)
            elif method == 'percentile':
                lower_bound = df[col].quantile(0.01)
                upper_bound = df[col].quantile(0.99)
                mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            
            df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def handle_missing_values(df, columns=None, strategy='mean'):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_filled = df.copy()
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
            elif strategy == 'ffill':
                df_filled[col] = df[col].ffill()
                continue
            elif strategy == 'bfill':
                df_filled[col] = df[col].bfill()
                continue
            
            df_filled[col] = df[col].fillna(fill_value)
    
    return df_filled

def clean_dataset(df, normalize_cols=None, outlier_cols=None, missing_cols=None, 
                  normalize_method='zscore', outlier_method='iqr', 
                  outlier_threshold=1.5, missing_strategy='mean'):
    
    df_processed = df.copy()
    
    df_processed = handle_missing_values(df_processed, missing_cols, missing_strategy)
    df_processed = remove_outliers(df_processed, outlier_cols, outlier_method, outlier_threshold)
    df_processed = normalize_data(df_processed, normalize_cols, normalize_method)
    
    return df_processed