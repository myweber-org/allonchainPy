
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    threshold (float): IQR multiplier (default 1.5)
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df.copy()

def normalize_minmax(dataframe, columns=None):
    """
    Normalize specified columns using min-max scaling.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    columns (list): List of column names to normalize (default: all numeric columns)
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    result_df = dataframe.copy()
    
    for col in columns:
        if col not in result_df.columns:
            continue
            
        if result_df[col].dtype in [np.float64, np.int64]:
            min_val = result_df[col].min()
            max_val = result_df[col].max()
            
            if max_val > min_val:
                result_df[col] = (result_df[col] - min_val) / (max_val - min_val)
    
    return result_df

def detect_skewed_columns(dataframe, skew_threshold=0.5):
    """
    Identify columns with significant skewness.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    skew_threshold (float): Absolute skewness threshold (default 0.5)
    
    Returns:
    dict: Dictionary with column names and their skewness values
    """
    skewed_cols = {}
    
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        skewness = stats.skew(dataframe[col].dropna())
        if abs(skewness) > skew_threshold:
            skewed_cols[col] = skewness
    
    return skewed_cols

def clean_dataset(dataframe, outlier_columns=None, normalize=True, skew_threshold=0.5):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    outlier_columns (list): Columns to process for outliers (default: all numeric)
    normalize (bool): Whether to normalize numeric columns
    skew_threshold (float): Skewness detection threshold
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary with cleaning statistics
    """
    if outlier_columns is None:
        outlier_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = dataframe.copy()
    stats_dict = {
        'original_rows': len(dataframe),
        'original_columns': len(dataframe.columns),
        'outlier_removal': {},
        'skewed_columns': {}
    }
    
    for col in outlier_columns:
        if col in cleaned_df.columns and cleaned_df[col].dtype in [np.float64, np.int64]:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed = original_count - len(cleaned_df)
            stats_dict['outlier_removal'][col] = removed
    
    if normalize:
        cleaned_df = normalize_minmax(cleaned_df)
        stats_dict['normalized'] = True
    
    skewed_cols = detect_skewed_columns(cleaned_df, skew_threshold)
    stats_dict['skewed_columns'] = skewed_cols
    stats_dict['final_rows'] = len(cleaned_df)
    
    return cleaned_df, stats_dict

def validate_dataframe(dataframe, required_columns=None, min_rows=10):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    dataframe (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(dataframe, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(dataframe) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in dataframe.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    if dataframe.isnull().all().any():
        return False, "Some columns contain only null values"
    
    return True, "DataFrame validation passed"