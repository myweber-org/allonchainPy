
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        threshold: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data
    
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        DataFrame with standardized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data
    
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values in numeric columns.
    
    Args:
        data: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', 'drop')
    
    Returns:
        DataFrame with handled missing values
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        return data.dropna(subset=numeric_cols)
    
    for col in numeric_cols:
        if data[col].isnull().any():
            if strategy == 'mean':
                fill_value = data[col].mean()
            elif strategy == 'median':
                fill_value = data[col].median()
            elif strategy == 'mode':
                fill_value = data[col].mode()[0]
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            data[col] = data[col].fillna(fill_value)
    
    return data

def clean_dataset(data, numeric_columns=None, outlier_threshold=1.5, 
                  normalize=True, standardize=False, missing_strategy='mean'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process (None for all numeric)
        outlier_threshold: IQR threshold for outlier removal
        normalize: whether to apply min-max normalization
        standardize: whether to apply z-score standardization
        missing_strategy: strategy for handling missing values
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    cleaned_data = handle_missing_values(cleaned_data, strategy=missing_strategy)
    
    for col in numeric_columns:
        if col in cleaned_data.columns:
            cleaned_data = remove_outliers_iqr(cleaned_data, col, outlier_threshold)
            
            if normalize:
                cleaned_data = normalize_minmax(cleaned_data, col)
            
            if standardize:
                cleaned_data = standardize_zscore(cleaned_data, col)
    
    return cleaned_data

def validate_data(data, check_duplicates=True, check_infinite=True):
    """
    Validate data quality.
    
    Args:
        data: pandas DataFrame
        check_duplicates: check for duplicate rows
        check_infinite: check for infinite values
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'missing_values': data.isnull().sum().to_dict(),
        'duplicate_rows': 0,
        'infinite_values': False
    }
    
    if check_duplicates:
        validation_results['duplicate_rows'] = data.duplicated().sum()
    
    if check_infinite:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if np.any(np.isinf(data[col])):
                validation_results['infinite_values'] = True
                break
    
    return validation_results
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    factor (float): Multiplier for IQR (default 1.5)
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)].copy()

def normalize_minmax(data, column):
    """
    Normalize a column using min-max scaling to [0, 1] range.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    result = data.copy()
    min_val = result[column].min()
    max_val = result[column].max()
    
    if max_val == min_val:
        result[column] = 0.5
    else:
        result[column] = (result[column] - min_val) / (max_val - min_val)
    
    return result

def standardize_zscore(data, column):
    """
    Standardize a column using z-score normalization.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to standardize
    
    Returns:
    pd.DataFrame: DataFrame with standardized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    result = data.copy()
    mean_val = result[column].mean()
    std_val = result[column].std()
    
    if std_val == 0:
        result[column] = 0
    else:
        result[column] = (result[column] - mean_val) / std_val
    
    return result

def clean_dataset(data, numeric_columns, outlier_factor=1.5, normalize=True):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names to process
    outlier_factor (float): Factor for IQR outlier detection
    normalize (bool): Whether to apply min-max normalization
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column in cleaned_data.columns:
            cleaned_data = remove_outliers_iqr(cleaned_data, column, outlier_factor)
            
            if normalize:
                cleaned_data = normalize_minmax(cleaned_data, column)
    
    return cleaned_data.reset_index(drop=True)

def validate_data(data, required_columns, numeric_columns):
    """
    Validate dataset structure and content.
    
    Parameters:
    data (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    numeric_columns (list): List of column names that should be numeric
    
    Returns:
    dict: Dictionary with validation results
    """
    validation_result = {
        'is_valid': True,
        'missing_columns': [],
        'non_numeric_columns': [],
        'null_counts': {}
    }
    
    for column in required_columns:
        if column not in data.columns:
            validation_result['missing_columns'].append(column)
            validation_result['is_valid'] = False
    
    for column in numeric_columns:
        if column in data.columns:
            if not pd.api.types.is_numeric_dtype(data[column]):
                validation_result['non_numeric_columns'].append(column)
                validation_result['is_valid'] = False
            
            null_count = data[column].isnull().sum()
            if null_count > 0:
                validation_result['null_counts'][column] = null_count
    
    return validation_result