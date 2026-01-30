import pandas as pd
import numpy as np
import re

def clean_column_names(df):
    """
    Standardize column names by converting to lowercase,
    replacing spaces with underscores, and removing special characters.
    """
    cleaned_columns = []
    for col in df.columns:
        col_str = str(col)
        col_str = col_str.lower().strip()
        col_str = re.sub(r'\s+', '_', col_str)
        col_str = re.sub(r'[^a-z0-9_]', '', col_str)
        cleaned_columns.append(col_str)
    df.columns = cleaned_columns
    return df

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in specified columns using different strategies.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_filled = df.copy()
    
    for col in columns:
        if col in df.columns:
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0] if not df[col].mode().empty else np.nan
            elif strategy == 'constant':
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            df_filled[col] = df[col].fillna(fill_value)
    
    return df_filled

def remove_outliers_iqr(df, columns=None, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
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

def standardize_numeric_columns(df, columns=None):
    """
    Standardize numeric columns to have zero mean and unit variance.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_standardized = df.copy()
    
    for col in columns:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                df_standardized[col] = (df[col] - mean_val) / std_val
    
    return df_standardized

def clean_data_pipeline(df, config=None):
    """
    Execute a complete data cleaning pipeline based on configuration.
    """
    if config is None:
        config = {
            'clean_columns': True,
            'handle_missing': True,
            'remove_outliers': True,
            'standardize': True
        }
    
    df_clean = df.copy()
    
    if config.get('clean_columns', False):
        df_clean = clean_column_names(df_clean)
    
    if config.get('handle_missing', False):
        df_clean = handle_missing_values(df_clean, strategy='mean')
    
    if config.get('remove_outliers', False):
        df_clean = remove_outliers_iqr(df_clean)
    
    if config.get('standardize', False):
        df_clean = standardize_numeric_columns(df_clean)
    
    return df_clean

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate dataframe structure and content.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if len(df) < min_rows:
        raise ValueError(f"DataFrame must have at least {min_rows} rows")
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True