
import pandas as pd
import numpy as np
from scipy import stats

def normalize_data(df, columns=None, method='zscore'):
    """
    Normalize specified columns in DataFrame.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to normalize (default: all numeric columns)
        method: normalization method ('zscore', 'minmax', 'robust')
    
    Returns:
        Normalized DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_normalized = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'zscore':
            df_normalized[col] = stats.zscore(df[col])
        elif method == 'minmax':
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max != col_min:
                df_normalized[col] = (df[col] - col_min) / (col_max - col_min)
        elif method == 'robust':
            median = df[col].median()
            iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
            if iqr != 0:
                df_normalized[col] = (df[col] - median) / iqr
    
    return df_normalized

def remove_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Remove outliers from specified columns.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to process (default: all numeric columns)
        method: outlier detection method ('iqr', 'zscore')
        threshold: threshold for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = df.copy()
    mask = pd.Series([True] * len(df))
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            col_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[col].fillna(df[col].mean())))
            col_mask = z_scores < threshold
        
        mask = mask & col_mask
    
    return df_clean[mask].reset_index(drop=True)

def handle_missing_values(df, columns=None, strategy='mean'):
    """
    Handle missing values in specified columns.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to process (default: all columns)
        strategy: imputation strategy ('mean', 'median', 'mode', 'drop')
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = df.columns.tolist()
    
    df_processed = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if strategy == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
            df_processed[col] = df[col].fillna(df[col].mean())
        elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[col]):
            df_processed[col] = df[col].fillna(df[col].median())
        elif strategy == 'mode':
            df_processed[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None)
        elif strategy == 'drop':
            df_processed = df_processed.dropna(subset=[col])
    
    return df_processed

def validate_dataframe(df, required_columns=None, numeric_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of required column names
        numeric_columns: list of columns that should be numeric
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    if numeric_columns:
        non_numeric = [col for col in numeric_columns if not pd.api.types.is_numeric_dtype(df[col])]
        if non_numeric:
            return False, f"Non-numeric columns that should be numeric: {non_numeric}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    return True, "DataFrame validation passed"import pandas as pd

def clean_dataset(df, columns_to_check=None, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Args:
        df: pandas DataFrame to clean.
        columns_to_check: List of columns to check for duplicates. If None, checks all columns.
        drop_duplicates: Boolean indicating whether to drop duplicate rows.
        fill_missing: Strategy to fill missing values. Options: 'mean', 'median', 'mode', or a scalar value.
    
    Returns:
        Cleaned pandas DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        if columns_to_check:
            cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check)
        else:
            cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        for column in cleaned_df.select_dtypes(include=['number']).columns:
            if cleaned_df[column].isnull().any():
                if fill_missing == 'mean':
                    fill_value = cleaned_df[column].mean()
                elif fill_missing == 'median':
                    fill_value = cleaned_df[column].median()
                elif fill_missing == 'mode':
                    fill_value = cleaned_df[column].mode()[0]
                else:
                    fill_value = fill_missing
                
                cleaned_df[column] = cleaned_df[column].fillna(fill_value)
    
    return cleaned_df

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
        df: pandas DataFrame.
        column: Column name to process.
        multiplier: IQR multiplier for outlier detection.
    
    Returns:
        DataFrame with outliers removed.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]