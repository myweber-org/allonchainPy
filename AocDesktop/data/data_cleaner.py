import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to process
    factor (float): Multiplier for IQR
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to normalize
    
    Returns:
    pd.DataFrame: Dataframe with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        data[f'{column}_normalized'] = 0.5
    else:
        data[f'{column}_normalized'] = (data[column] - min_val) / (max_val - min_val)
    
    return data

def z_score_normalize(data, column):
    """
    Normalize data using Z-score method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to normalize
    
    Returns:
    pd.DataFrame: Dataframe with z-score normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        data[f'{column}_zscore'] = 0
    else:
        data[f'{column}_zscore'] = (data[column] - mean_val) / std_val
    
    return data

def clean_dataset(data, numeric_columns, outlier_factor=1.5):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    numeric_columns (list): List of numeric column names to clean
    outlier_factor (float): Factor for IQR outlier detection
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column in cleaned_data.columns:
            # Remove outliers
            cleaned_data = remove_outliers_iqr(cleaned_data, column, outlier_factor)
            
            # Apply min-max normalization
            cleaned_data = normalize_minmax(cleaned_data, column)
            
            # Apply z-score normalization
            cleaned_data = z_score_normalize(cleaned_data, column)
    
    return cleaned_data

def validate_data(data, required_columns):
    """
    Validate that dataframe contains required columns and no null values.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    required_columns (list): List of required column names
    
    Returns:
    bool: True if validation passes, raises exception otherwise
    """
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if data.isnull().any().any():
        null_counts = data.isnull().sum()
        null_columns = null_counts[null_counts > 0].index.tolist()
        raise ValueError(f"Data contains null values in columns: {null_columns}")
    
    return True