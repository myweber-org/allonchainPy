import pandas as pd
import numpy as np

def load_and_clean_csv(filepath, drop_na=True, fill_strategy='mean'):
    """
    Load a CSV file and perform basic cleaning operations.
    
    Args:
        filepath (str): Path to the CSV file.
        drop_na (bool): If True, drop rows with missing values.
        fill_strategy (str): Strategy to fill missing values if drop_na is False.
                             Options: 'mean', 'median', 'mode', or 'zero'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if df.empty:
        return df
    
    if drop_na:
        df_cleaned = df.dropna()
    else:
        df_cleaned = df.copy()
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df_cleaned[col].isnull().any():
                if fill_strategy == 'mean':
                    fill_value = df_cleaned[col].mean()
                elif fill_strategy == 'median':
                    fill_value = df_cleaned[col].median()
                elif fill_strategy == 'mode':
                    fill_value = df_cleaned[col].mode()[0]
                elif fill_strategy == 'zero':
                    fill_value = 0
                else:
                    raise ValueError(f"Unsupported fill strategy: {fill_strategy}")
                
                df_cleaned[col].fillna(fill_value, inplace=True)
        
        object_cols = df_cleaned.select_dtypes(include=['object']).columns
        for col in object_cols:
            df_cleaned[col].fillna('Unknown', inplace=True)
    
    return df_cleaned

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list): Columns to consider for identifying duplicates.
        keep (str): Which duplicates to keep. Options: 'first', 'last', False.
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def normalize_numeric_columns(df, columns=None, method='minmax'):
    """
    Normalize numeric columns to a common scale.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): Columns to normalize. If None, normalize all numeric columns.
        method (str): Normalization method. Options: 'minmax', 'zscore'.
    
    Returns:
        pd.DataFrame: DataFrame with normalized columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'minmax':
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max != col_min:
                df_normalized[col] = (df[col] - col_min) / (col_max - col_min)
            else:
                df_normalized[col] = 0
        
        elif method == 'zscore':
            col_mean = df[col].mean()
            col_std = df[col].std()
            if col_std != 0:
                df_normalized[col] = (df[col] - col_mean) / col_std
            else:
                df_normalized[col] = 0
    
    return df_normalized

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of columns that must be present.
        min_rows (int): Minimum number of rows required.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"