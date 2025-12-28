import numpy as np
import pandas as pd

def remove_outliers_iqr(df, columns, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to process
    factor (float): Multiplier for IQR (default 1.5)
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
            df_clean = df_clean[mask]
    return df_clean.reset_index(drop=True)

def normalize_minmax(df, columns):
    """
    Normalize columns using Min-Max scaling.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to normalize
    
    Returns:
    pd.DataFrame: Dataframe with normalized columns
    """
    df_norm = df.copy()
    for col in columns:
        if col in df_norm.columns:
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            if max_val > min_val:
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
    return df_norm

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in dataframe columns.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    strategy (str): Imputation strategy ('mean', 'median', 'mode', 'drop')
    columns (list): List of columns to process (None for all numeric columns)
    
    Returns:
    pd.DataFrame: Dataframe with handled missing values
    """
    df_processed = df.copy()
    
    if columns is None:
        columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col in df_processed.columns:
            if strategy == 'mean':
                df_processed[col].fillna(df_processed[col].mean(), inplace=True)
            elif strategy == 'median':
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
            elif strategy == 'mode':
                df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                df_processed = df_processed.dropna(subset=[col])
    
    return df_processed.reset_index(drop=True) if strategy == 'drop' else df_processed

def clean_dataset(df, numeric_columns, outlier_factor=1.5, normalize=True, missing_strategy='mean'):
    """
    Complete data cleaning pipeline.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    numeric_columns (list): List of numeric column names to process
    outlier_factor (float): IQR factor for outlier removal
    normalize (bool): Whether to apply min-max normalization
    missing_strategy (str): Strategy for handling missing values
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    if not numeric_columns:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = df.copy()
    
    df_clean = handle_missing_values(df_clean, strategy=missing_strategy, columns=numeric_columns)
    df_clean = remove_outliers_iqr(df_clean, numeric_columns, factor=outlier_factor)
    
    if normalize:
        df_clean = normalize_minmax(df_clean, numeric_columns)
    
    return df_clean