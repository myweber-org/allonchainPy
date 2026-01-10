
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        dataframe: pandas DataFrame
        column: column name to process
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    return filtered_df

def normalize_minmax(dataframe, columns=None):
    """
    Normalize specified columns using Min-Max scaling.
    
    Args:
        dataframe: pandas DataFrame
        columns: list of column names to normalize (default: all numeric columns)
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns
    
    normalized_df = dataframe.copy()
    for col in columns:
        if col in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[col]):
            col_min = dataframe[col].min()
            col_max = dataframe[col].max()
            if col_max != col_min:
                normalized_df[col] = (dataframe[col] - col_min) / (col_max - col_min)
            else:
                normalized_df[col] = 0
    
    return normalized_df

def z_score_normalize(dataframe, columns=None):
    """
    Normalize specified columns using Z-score normalization.
    
    Args:
        dataframe: pandas DataFrame
        columns: list of column names to normalize (default: all numeric columns)
    
    Returns:
        DataFrame with Z-score normalized columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns
    
    normalized_df = dataframe.copy()
    for col in columns:
        if col in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[col]):
            mean_val = dataframe[col].mean()
            std_val = dataframe[col].std()
            if std_val > 0:
                normalized_df[col] = (dataframe[col] - mean_val) / std_val
            else:
                normalized_df[col] = 0
    
    return normalized_df

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Args:
        dataframe: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', 'drop')
        columns: list of column names to process (default: all columns)
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = dataframe.columns
    
    processed_df = dataframe.copy()
    
    for col in columns:
        if col not in processed_df.columns:
            continue
            
        if processed_df[col].isnull().any():
            if strategy == 'drop':
                processed_df = processed_df.dropna(subset=[col])
            elif strategy == 'mean' and pd.api.types.is_numeric_dtype(processed_df[col]):
                processed_df[col] = processed_df[col].fillna(processed_df[col].mean())
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(processed_df[col]):
                processed_df[col] = processed_df[col].fillna(processed_df[col].median())
            elif strategy == 'mode':
                mode_val = processed_df[col].mode()
                if not mode_val.empty:
                    processed_df[col] = processed_df[col].fillna(mode_val[0])
    
    return processed_df

def validate_dataframe(dataframe, required_columns=None, numeric_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        dataframe: pandas DataFrame to validate
        required_columns: list of required column names
        numeric_columns: list of columns that should be numeric
    
    Returns:
        tuple: (is_valid, validation_errors)
    """
    errors = []
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in dataframe.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
    
    if numeric_columns:
        non_numeric = [col for col in numeric_columns 
                      if col in dataframe.columns 
                      and not pd.api.types.is_numeric_dtype(dataframe[col])]
        if non_numeric:
            errors.append(f"Non-numeric columns that should be numeric: {non_numeric}")
    
    if dataframe.empty:
        errors.append("DataFrame is empty")
    
    return len(errors) == 0, errors