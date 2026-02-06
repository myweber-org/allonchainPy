
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, columns=None):
    """
    Remove outliers from DataFrame using Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to process. If None, process all numeric columns.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize specified columns in DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to normalize
    method (str): Normalization method - 'minmax' or 'zscore'
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_norm = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val != min_val:
                df_norm[col] = (df[col] - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val != 0:
                df_norm[col] = (df[col] - mean_val) / std_val
    
    return df_norm

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Strategy for handling missing values - 'mean', 'median', 'mode', or 'drop'
    columns (list): List of column names to process
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    if columns is None:
        columns = df.columns.tolist()
    
    df_handled = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if strategy == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
            df_handled[col].fillna(df[col].mean(), inplace=True)
        
        elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[col]):
            df_handled[col].fillna(df[col].median(), inplace=True)
        
        elif strategy == 'mode':
            df_handled[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None, inplace=True)
        
        elif strategy == 'drop':
            df_handled = df_handled.dropna(subset=[col])
    
    return df_handled.reset_index(drop=True)