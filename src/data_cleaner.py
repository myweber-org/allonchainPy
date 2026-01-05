
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns, factor=1.5):
    """
    Remove outliers using IQR method
    """
    df_clean = df.copy()
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def normalize_minmax(df, columns):
    """
    Normalize data using min-max scaling
    """
    df_norm = df.copy()
    for col in columns:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val != min_val:
                df_norm[col] = (df[col] - min_val) / (max_val - min_val)
    return df_norm

def standardize_zscore(df, columns):
    """
    Standardize data using z-score
    """
    df_std = df.copy()
    for col in columns:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                df_std[col] = (df[col] - mean_val) / std_val
    return df_std

def clean_dataset(df, numeric_columns, outlier_method='iqr', scaling_method='standardize'):
    """
    Main cleaning pipeline
    """
    if outlier_method == 'iqr':
        df_clean = remove_outliers_iqr(df, numeric_columns)
    elif outlier_method == 'zscore':
        z_scores = np.abs(stats.zscore(df[numeric_columns]))
        df_clean = df[(z_scores < 3).all(axis=1)]
    else:
        df_clean = df.copy()
    
    if scaling_method == 'normalize':
        df_final = normalize_minmax(df_clean, numeric_columns)
    elif scaling_method == 'standardize':
        df_final = standardize_zscore(df_clean, numeric_columns)
    else:
        df_final = df_clean
    
    return df_final

def validate_data(df, required_columns, min_rows=10):
    """
    Validate dataset structure and content
    """
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if len(df) < min_rows:
        raise ValueError(f"Dataset has fewer than {min_rows} rows")
    
    if df.isnull().sum().sum() > 0:
        print("Warning: Dataset contains missing values")
    
    return True