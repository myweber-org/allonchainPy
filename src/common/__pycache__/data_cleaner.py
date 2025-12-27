
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    threshold (float): IQR multiplier for outlier detection
    
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
    columns (list): List of column names to normalize. If None, normalize all numeric columns.
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    if columns is None:
        numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col not in normalized_df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        
        if not np.issubdtype(normalized_df[col].dtype, np.number):
            raise ValueError(f"Column '{col}' is not numeric")
        
        col_min = normalized_df[col].min()
        col_max = normalized_df[col].max()
        
        if col_max != col_min:
            normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
        else:
            normalized_df[col] = 0
    
    return normalized_df

def zscore_normalize(dataframe, columns=None, threshold=3):
    """
    Normalize using z-score and optionally remove extreme outliers.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    columns (list): List of column names to process
    threshold (float): Z-score threshold for outlier removal
    
    Returns:
    pd.DataFrame: Processed DataFrame
    """
    if columns is None:
        numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    processed_df = dataframe.copy()
    
    for col in columns:
        if col not in processed_df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        
        if not np.issubdtype(processed_df[col].dtype, np.number):
            raise ValueError(f"Column '{col}' is not numeric")
        
        z_scores = np.abs(stats.zscore(processed_df[col], nan_policy='omit'))
        mask = z_scores <= threshold
        
        col_mean = processed_df.loc[mask, col].mean()
        col_std = processed_df.loc[mask, col].std()
        
        if col_std > 0:
            processed_df[col] = (processed_df[col] - col_mean) / col_std
        else:
            processed_df[col] = 0
    
    return processed_df

def clean_dataset(dataframe, numeric_columns=None, outlier_threshold=1.5, normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric columns to process
    outlier_threshold (float): IQR threshold for outlier removal
    normalize_method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = dataframe.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_threshold)
    
    if normalize_method == 'minmax':
        cleaned_df = normalize_minmax(cleaned_df, numeric_columns)
    elif normalize_method == 'zscore':
        cleaned_df = zscore_normalize(cleaned_df, numeric_columns)
    else:
        raise ValueError("normalize_method must be 'minmax' or 'zscore'")
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(dataframe, required_columns=None, min_rows=1):
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
        return False, f"DataFrame must have at least {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in dataframe.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"