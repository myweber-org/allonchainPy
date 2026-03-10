
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    cleaned_df = df.copy()
    for col in columns:
        if col in cleaned_df.columns:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    return cleaned_df

def normalize_minmax(df, columns):
    normalized_df = df.copy()
    for col in columns:
        if col in normalized_df.columns:
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            if max_val != min_val:
                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
    return normalized_df

def handle_missing_values(df, strategy='mean'):
    handled_df = df.copy()
    numeric_cols = handled_df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        for col in numeric_cols:
            handled_df[col].fillna(handled_df[col].mean(), inplace=True)
    elif strategy == 'median':
        for col in numeric_cols:
            handled_df[col].fillna(handled_df[col].median(), inplace=True)
    elif strategy == 'mode':
        for col in numeric_cols:
            handled_df[col].fillna(handled_df[col].mode()[0], inplace=True)
    
    return handled_df

def clean_dataset(df, numeric_columns, outlier_removal=True, normalization=True, missing_strategy='mean'):
    df_cleaned = df.copy()
    
    if outlier_removal:
        df_cleaned = remove_outliers_iqr(df_cleaned, numeric_columns)
    
    df_cleaned = handle_missing_values(df_cleaned, strategy=missing_strategy)
    
    if normalization:
        df_cleaned = normalize_minmax(df_cleaned, numeric_columns)
    
    return df_cleaned