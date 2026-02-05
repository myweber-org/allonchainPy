import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns):
    """
    Remove outliers using the Interquartile Range method.
    Returns a cleaned DataFrame.
    """
    df_clean = df.copy()
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean.reset_index(drop=True)

def remove_outliers_zscore(df, columns, threshold=3):
    """
    Remove outliers using Z-score method.
    Returns a cleaned DataFrame.
    """
    df_clean = df.copy()
    for col in columns:
        if col in df.columns:
            z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
            df_clean = df_clean[(z_scores < threshold) | (df_clean[col].isna())]
    return df_clean.reset_index(drop=True)

def normalize_minmax(df, columns):
    """
    Normalize specified columns using Min-Max scaling.
    Returns DataFrame with normalized columns.
    """
    df_norm = df.copy()
    for col in columns:
        if col in df.columns:
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            if max_val != min_val:
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
    return df_norm

def normalize_zscore(df, columns):
    """
    Normalize specified columns using Z-score standardization.
    Returns DataFrame with normalized columns.
    """
    df_norm = df.copy()
    for col in columns:
        if col in df.columns:
            mean_val = df_norm[col].mean()
            std_val = df_norm[col].std()
            if std_val > 0:
                df_norm[col] = (df_norm[col] - mean_val) / std_val
    return df_norm

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy.
    Supported strategies: 'mean', 'median', 'mode', 'drop'
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.columns
    
    numeric_cols = df_clean[columns].select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        df_clean = df_clean.dropna(subset=numeric_cols)
    elif strategy == 'mean':
        for col in numeric_cols:
            df_clean[col].fillna(df_clean[col].mean(), inplace=True)
    elif strategy == 'median':
        for col in numeric_cols:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    elif strategy == 'mode':
        for col in numeric_cols:
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    
    return df_clean.reset_index(drop=True)

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    Returns tuple of (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"