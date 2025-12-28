import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns=None, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of columns to process, if None processes all numeric columns
    factor (float): Multiplier for IQR (default 1.5)
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def remove_outliers_zscore(df, columns=None, threshold=3):
    """
    Remove outliers using Z-score method.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of columns to process
    threshold (float): Z-score threshold (default 3)
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    for col in columns:
        if col in df.columns:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            mask = z_scores < threshold
            df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_minmax(df, columns=None, feature_range=(0, 1)):
    """
    Normalize data using Min-Max scaling.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of columns to normalize
    feature_range (tuple): Desired range of transformed data
    
    Returns:
    pd.DataFrame: Normalized dataframe
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_norm = df.copy()
    min_val, max_val = feature_range
    
    for col in columns:
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_max - col_min != 0:
                df_norm[col] = (df[col] - col_min) / (col_max - col_min)
                df_norm[col] = df_norm[col] * (max_val - min_val) + min_val
    
    return df_norm

def normalize_zscore(df, columns=None):
    """
    Normalize data using Z-score standardization.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of columns to normalize
    
    Returns:
    pd.DataFrame: Standardized dataframe
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_std = df.copy()
    
    for col in columns:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            
            if std_val != 0:
                df_std[col] = (df[col] - mean_val) / std_val
    
    return df_std

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in dataframe.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    strategy (str): Imputation strategy ('mean', 'median', 'mode', 'drop')
    columns (list): List of columns to process
    
    Returns:
    pd.DataFrame: Dataframe with handled missing values
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_processed = df.copy()
    
    if strategy == 'drop':
        return df_processed.dropna(subset=columns)
    
    for col in columns:
        if col in df.columns:
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
            else:
                fill_value = 0
            
            df_processed[col] = df[col].fillna(fill_value)
    
    return df_processed

def clean_data_pipeline(df, outlier_method='iqr', normalize_method='minmax', 
                       missing_strategy='mean', outlier_threshold=3):
    """
    Complete data cleaning pipeline.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    outlier_method (str): Outlier removal method ('iqr', 'zscore', or None)
    normalize_method (str): Normalization method ('minmax', 'zscore', or None)
    missing_strategy (str): Missing value strategy
    outlier_threshold (float): Threshold for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    df_clean = df.copy()
    
    # Handle missing values first
    df_clean = handle_missing_values(df_clean, strategy=missing_strategy)
    
    # Remove outliers if specified
    if outlier_method == 'iqr':
        df_clean = remove_outliers_iqr(df_clean)
    elif outlier_method == 'zscore':
        df_clean = remove_outliers_zscore(df_clean, threshold=outlier_threshold)
    
    # Normalize data if specified
    if normalize_method == 'minmax':
        df_clean = normalize_minmax(df_clean)
    elif normalize_method == 'zscore':
        df_clean = normalize_zscore(df_clean)
    
    return df_clean