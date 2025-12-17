
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
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

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (bool, str) indicating validation success and message.
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"import numpy as np
import pandas as pd
from scipy import stats

def normalize_data(data, method='zscore'):
    if method == 'zscore':
        return (data - np.mean(data)) / np.std(data)
    elif method == 'minmax':
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    else:
        raise ValueError("Method must be 'zscore' or 'minmax'")

def remove_outliers_iqr(data, threshold=1.5):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return data[(data >= lower_bound) & (data <= upper_bound)]

def clean_dataset(df, column, normalize_method='zscore', outlier_threshold=1.5):
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in DataFrame")
    
    original_data = df[column].dropna().values
    cleaned_data = remove_outliers_iqr(original_data, outlier_threshold)
    normalized_data = normalize_data(cleaned_data, normalize_method)
    
    result_df = pd.DataFrame({
        'original': original_data,
        'cleaned': np.concatenate([cleaned_data, [np.nan] * (len(original_data) - len(cleaned_data))]),
        'normalized': np.concatenate([normalized_data, [np.nan] * (len(original_data) - len(normalized_data))])
    })
    
    return result_df

def calculate_statistics(data):
    return {
        'mean': np.mean(data),
        'std': np.std(data),
        'median': np.median(data),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data)
    }