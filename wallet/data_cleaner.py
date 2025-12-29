import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns=None, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to process (default: all numeric columns)
        factor: multiplier for IQR (default: 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def zscore_normalize(df, columns=None, threshold=3):
    """
    Normalize data using Z-score and optionally remove extreme outliers.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to normalize (default: all numeric columns)
        threshold: Z-score threshold for outlier removal (default: 3)
    
    Returns:
        Normalized DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    for col in columns:
        z_scores = np.abs(stats.zscore(df_normalized[col].dropna()))
        
        if threshold is not None:
            outlier_mask = z_scores < threshold
            valid_indices = df_normalized[col].dropna().index[outlier_mask]
            df_normalized.loc[~df_normalized.index.isin(valid_indices), col] = np.nan
        
        mean_val = df_normalized[col].mean()
        std_val = df_normalized[col].std()
        
        if std_val > 0:
            df_normalized[col] = (df_normalized[col] - mean_val) / std_val
    
    return df_normalized

def minmax_scale(df, columns=None, feature_range=(0, 1)):
    """
    Scale data to a specified range using Min-Max scaling.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to scale (default: all numeric columns)
        feature_range: tuple of (min, max) for scaled range
    
    Returns:
        Scaled DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_scaled = df.copy()
    min_val, max_val = feature_range
    
    for col in columns:
        col_min = df_scaled[col].min()
        col_max = df_scaled[col].max()
        col_range = col_max - col_min
        
        if col_range > 0:
            df_scaled[col] = min_val + (df_scaled[col] - col_min) * (max_val - min_val) / col_range
    
    return df_scaled

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        df: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', 'constant', 'drop')
        columns: list of column names to process (default: all columns)
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = df.columns
    
    df_handled = df.copy()
    
    for col in columns:
        if df_handled[col].isnull().any():
            if strategy == 'mean':
                fill_value = df_handled[col].mean()
            elif strategy == 'median':
                fill_value = df_handled[col].median()
            elif strategy == 'mode':
                fill_value = df_handled[col].mode()[0]
            elif strategy == 'constant':
                fill_value = 0
            elif strategy == 'drop':
                df_handled = df_handled.dropna(subset=[col])
                continue
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            df_handled[col] = df_handled[col].fillna(fill_value)
    
    return df_handled.reset_index(drop=True)