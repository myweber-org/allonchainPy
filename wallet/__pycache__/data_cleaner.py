
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns):
    cleaned_df = df.copy()
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            cleaned_df = cleaned_df[mask]
    return cleaned_df.reset_index(drop=True)

def remove_outliers_zscore(df, columns, threshold=3):
    cleaned_df = df.copy()
    for col in columns:
        if col in df.columns:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            mask = z_scores < threshold
            valid_indices = df[col].dropna().index[mask]
            cleaned_df = cleaned_df.loc[valid_indices.union(df[df[col].isna()].index)]
    return cleaned_df.reset_index(drop=True)

def normalize_minmax(df, columns):
    normalized_df = df.copy()
    for col in columns:
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max != col_min:
                normalized_df[col] = (df[col] - col_min) / (col_max - col_min)
            else:
                normalized_df[col] = 0
    return normalized_df

def normalize_zscore(df, columns):
    normalized_df = df.copy()
    for col in columns:
        if col in df.columns:
            col_mean = df[col].mean()
            col_std = df[col].std()
            if col_std > 0:
                normalized_df[col] = (df[col] - col_mean) / col_std
            else:
                normalized_df[col] = 0
    return normalized_df

def handle_missing_values(df, strategy='mean', columns=None):
    processed_df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
            elif strategy == 'zero':
                fill_value = 0
            else:
                fill_value = df[col].mean()
            
            processed_df[col] = df[col].fillna(fill_value)
    
    return processed_df

def get_data_summary(df):
    summary = {
        'original_shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
    }
    return summary

def clean_dataset(df, numeric_columns=None, outlier_method='iqr', normalize_method=None, missing_strategy='mean'):
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy, columns=numeric_columns)
    
    if outlier_method == 'iqr':
        cleaned_df = remove_outliers_iqr(cleaned_df, numeric_columns)
    elif outlier_method == 'zscore':
        cleaned_df = remove_outliers_zscore(cleaned_df, numeric_columns)
    
    if normalize_method == 'minmax':
        cleaned_df = normalize_minmax(cleaned_df, numeric_columns)
    elif normalize_method == 'zscore':
        cleaned_df = normalize_zscore(cleaned_df, numeric_columns)
    
    return cleaned_df