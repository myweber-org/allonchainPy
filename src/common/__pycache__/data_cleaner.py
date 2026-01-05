import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using the Interquartile Range method.
    Returns a boolean mask for outliers.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    return (data[column] < lower_bound) | (data[column] > upper_bound)

def remove_outliers(df, columns, threshold=1.5):
    """
    Remove outliers from specified columns using IQR method.
    Returns cleaned DataFrame.
    """
    df_clean = df.copy()
    for col in columns:
        outliers = detect_outliers_iqr(df_clean, col, threshold)
        df_clean = df_clean[~outliers]
    return df_clean.reset_index(drop=True)

def normalize_minmax(data, columns):
    """
    Apply min-max normalization to specified columns.
    Returns DataFrame with normalized columns.
    """
    df_normalized = data.copy()
    for col in columns:
        min_val = df_normalized[col].min()
        max_val = df_normalized[col].max()
        if max_val > min_val:
            df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
    return df_normalized

def standardize_zscore(data, columns):
    """
    Apply z-score standardization to specified columns.
    Returns DataFrame with standardized columns.
    """
    df_standardized = data.copy()
    for col in columns:
        mean_val = df_standardized[col].mean()
        std_val = df_standardized[col].std()
        if std_val > 0:
            df_standardized[col] = (df_standardized[col] - mean_val) / std_val
    return df_standardized

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy.
    Supported strategies: 'mean', 'median', 'mode', 'drop'
    """
    df_processed = df.copy()
    
    if columns is None:
        columns = df_processed.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        return df_processed.dropna(subset=columns)
    
    for col in columns:
        if df_processed[col].isnull().any():
            if strategy == 'mean':
                fill_value = df_processed[col].mean()
            elif strategy == 'median':
                fill_value = df_processed[col].median()
            elif strategy == 'mode':
                fill_value = df_processed[col].mode()[0]
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")
            
            df_processed[col] = df_processed[col].fillna(fill_value)
    
    return df_processed

def validate_dataframe(df, required_columns=None, numeric_columns=None):
    """
    Basic DataFrame validation.
    Returns tuple of (is_valid, error_message)
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    if numeric_columns:
        non_numeric = [col for col in numeric_columns if not np.issubdtype(df[col].dtype, np.number)]
        if non_numeric:
            return False, f"Non-numeric columns specified as numeric: {non_numeric}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    return True, "DataFrame validation passed"