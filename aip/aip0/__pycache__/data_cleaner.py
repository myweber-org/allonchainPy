import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns=None, threshold=1.5):
    """
    Remove outliers using Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of columns to process, None for all numeric columns
    threshold (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
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
        
        mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_minmax(df, columns=None):
    """
    Normalize data using Min-Max scaling.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of columns to normalize, None for all numeric columns
    
    Returns:
    pd.DataFrame: Dataframe with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    for col in columns:
        min_val = df_normalized[col].min()
        max_val = df_normalized[col].max()
        
        if max_val > min_val:
            df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
    
    return df_normalized

def detect_skewed_columns(df, threshold=0.5):
    """
    Detect skewed columns using skewness coefficient.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    threshold (float): Absolute skewness threshold
    
    Returns:
    list: Columns with absolute skewness greater than threshold
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    skewed_cols = []
    
    for col in numeric_cols:
        skewness = stats.skew(df[col].dropna())
        if abs(skewness) > threshold:
            skewed_cols.append((col, skewness))
    
    return sorted(skewed_cols, key=lambda x: abs(x[1]), reverse=True)

def log_transform_skewed(df, skewed_cols):
    """
    Apply log transformation to reduce skewness.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    skewed_cols (list): List of column names to transform
    
    Returns:
    pd.DataFrame: Dataframe with transformed columns
    """
    df_transformed = df.copy()
    
    for col in skewed_cols:
        if col in df.columns:
            min_val = df[col].min()
            if min_val <= 0:
                df_transformed[col] = np.log1p(df[col] - min_val + 1)
            else:
                df_transformed[col] = np.log(df[col])
    
    return df_transformed

def clean_dataset(df, outlier_threshold=1.5, skew_threshold=0.5):
    """
    Complete data cleaning pipeline.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    outlier_threshold (float): IQR threshold for outlier removal
    skew_threshold (float): Skewness detection threshold
    
    Returns:
    pd.DataFrame: Cleaned and normalized dataframe
    """
    print(f"Original shape: {df.shape}")
    
    df_clean = remove_outliers_iqr(df, threshold=outlier_threshold)
    print(f"After outlier removal: {df_clean.shape}")
    
    skewed = detect_skewed_columns(df_clean, threshold=skew_threshold)
    skewed_cols = [col for col, _ in skewed]
    
    if skewed_cols:
        print(f"Skewed columns detected: {skewed_cols}")
        df_clean = log_transform_skewed(df_clean, skewed_cols)
    
    df_normalized = normalize_minmax(df_clean)
    print("Data normalization completed")
    
    return df_normalized