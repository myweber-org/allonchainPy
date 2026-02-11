
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        dataframe: pandas DataFrame
        column: column name to process
        multiplier: IQR multiplier for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    return dataframe[(dataframe[column] >= lower_bound) & 
                     (dataframe[column] <= upper_bound)]

def normalize_minmax(dataframe, columns=None):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        dataframe: pandas DataFrame
        columns: list of columns to normalize (None for all numeric columns)
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns
    
    result = dataframe.copy()
    for col in columns:
        col_min = result[col].min()
        col_max = result[col].max()
        if col_max != col_min:
            result[col] = (result[col] - col_min) / (col_max - col_min)
    
    return result

def standardize_zscore(dataframe, columns=None):
    """
    Standardize data using Z-score normalization.
    
    Args:
        dataframe: pandas DataFrame
        columns: list of columns to standardize
    
    Returns:
        DataFrame with standardized columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns
    
    result = dataframe.copy()
    for col in columns:
        result[col] = stats.zscore(result[col], nan_policy='omit')
    
    return result

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        dataframe: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'drop'
        columns: list of columns to process
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = dataframe.columns
    
    result = dataframe.copy()
    
    for col in columns:
        if result[col].isnull().any():
            if strategy == 'mean':
                fill_value = result[col].mean()
            elif strategy == 'median':
                fill_value = result[col].median()
            elif strategy == 'mode':
                fill_value = result[col].mode()[0]
            elif strategy == 'drop':
                result = result.dropna(subset=[col])
                continue
            else:
                raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")
            
            result[col] = result[col].fillna(fill_value)
    
    return result

def clean_dataset(dataframe, outlier_columns=None, normalize=True, standardize=False, 
                  missing_strategy='mean'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        dataframe: pandas DataFrame
        outlier_columns: columns for outlier removal
        normalize: whether to apply min-max normalization
        standardize: whether to apply z-score standardization
        missing_strategy: strategy for handling missing values
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = dataframe.copy()
    
    # Handle missing values
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    # Remove outliers
    if outlier_columns:
        for col in outlier_columns:
            if col in cleaned_df.columns:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    # Apply normalization
    if normalize:
        cleaned_df = normalize_minmax(cleaned_df)
    
    # Apply standardization
    if standardize:
        cleaned_df = standardize_zscore(cleaned_df)
    
    return cleaned_df