import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Method to fill missing values. Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            elif fill_missing == 'median':
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None, inplace=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for required columns and data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names. Default is None.
    
    Returns:
    tuple: (bool, str) indicating success and message.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    return True, "DataFrame validation passed"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 30, 40, 50],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataframe(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B'])
    print(f"\nValidation: {is_valid}, Message: {message}")
import pandas as pd
import numpy as np

def remove_missing_rows(df, columns=None):
    if columns is None:
        columns = df.columns
    return df.dropna(subset=columns)

def fill_missing_with_mean(df, columns=None):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    df_filled = df.copy()
    for col in columns:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            df_filled[col] = df[col].fillna(df[col].mean())
    return df_filled

def remove_outliers_iqr(df, columns=None, multiplier=1.5):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    df_clean = df.copy()
    for col in columns:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def standardize_columns(df, columns=None):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    df_standardized = df.copy()
    for col in columns:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df_standardized[col] = (df[col] - mean) / std
    return df_standardizedimport numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, threshold=1.5):
    """
    Detect outliers using Interquartile Range method.
    
    Args:
        data: Array-like data
        threshold: IQR multiplier (default 1.5)
    
    Returns:
        Boolean mask of outliers
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return (data < lower_bound) | (data > upper_bound)

def winsorize_data(data, limits=(0.05, 0.05)):
    """
    Apply winsorization to limit extreme values.
    
    Args:
        data: Array-like data
        limits: Tuple of lower and upper limits
    
    Returns:
        Winsorized data
    """
    return stats.mstats.winsorize(data, limits=limits)

def normalize_minmax(data):
    """
    Normalize data to [0, 1] range using min-max scaling.
    
    Args:
        data: Array-like data
    
    Returns:
        Normalized data
    """
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max == data_min:
        return np.zeros_like(data)
    return (data - data_min) / (data_max - data_min)

def clean_dataframe(df, columns=None, outlier_method='iqr', normalize=False):
    """
    Clean DataFrame by handling outliers and optionally normalizing.
    
    Args:
        df: Input DataFrame
        columns: List of columns to process (default: all numeric columns)
        outlier_method: 'iqr' or 'winsorize'
        normalize: Whether to apply min-max normalization
    
    Returns:
        Cleaned DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    result_df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        col_data = df[col].values
        
        if outlier_method == 'iqr':
            outliers = detect_outliers_iqr(col_data)
            if np.any(outliers):
                col_median = np.median(col_data[~outliers])
                result_df.loc[outliers, col] = col_median
        elif outlier_method == 'winsorize':
            result_df[col] = winsorize_data(col_data)
        
        if normalize:
            result_df[col] = normalize_minmax(result_df[col].values)
    
    return result_df

def calculate_statistics(df):
    """
    Calculate basic statistics for numeric columns.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Dictionary of statistics
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats_dict = {}
    
    for col in numeric_cols:
        col_data = df[col].dropna()
        stats_dict[col] = {
            'mean': np.mean(col_data),
            'median': np.median(col_data),
            'std': np.std(col_data),
            'min': np.min(col_data),
            'max': np.max(col_data),
            'count': len(col_data),
            'missing': df[col].isna().sum()
        }
    
    return stats_dict