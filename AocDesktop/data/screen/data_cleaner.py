
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a specified column in a DataFrame using the IQR method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing statistical measures.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def clean_missing_values(df, strategy='drop'):
    """
    Handle missing values in a DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    strategy (str): Strategy to handle missing values ('drop' or 'fill').
    
    Returns:
    pd.DataFrame: DataFrame with missing values handled.
    """
    if strategy == 'drop':
        cleaned_df = df.dropna()
    elif strategy == 'fill':
        cleaned_df = df.fillna(df.mean())
    else:
        raise ValueError("Strategy must be either 'drop' or 'fill'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 100, 12, 11, 10, 9, 8, 12, 13, 14]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("DataFrame after outlier removal:")
    print(cleaned_df)
    print()
    
    stats = calculate_basic_stats(cleaned_df, 'values')
    print("Basic statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    if max_val - min_val == 0:
        return df[column]
    return (df[column] - min_val) / (max_val - min_val)

def standardize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    if std_val == 0:
        return df[column]
    return (df[column] - mean_val) / std_val

def clean_dataset(df, numeric_columns, outlier_removal=True, normalization='standard'):
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col not in cleaned_df.columns:
            continue
            
        if outlier_removal:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        
        if normalization == 'minmax':
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
        elif normalization == 'standard':
            cleaned_df[col] = standardize_zscore(cleaned_df, col)
    
    return cleaned_df.reset_index(drop=True)

def validate_data(df, required_columns, allow_nan=False):
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if not allow_nan:
        if df.isnull().any().any():
            raise ValueError("Dataset contains NaN values")
    
    return Trueimport numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns=None, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to process (default: all numeric columns)
        factor: multiplier for IQR (default: 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
        df_clean = df_clean[mask]
    
    return df_clean

def normalize_minmax(df, columns=None, feature_range=(0, 1)):
    """
    Normalize data using min-max scaling.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to normalize (default: all numeric columns)
        feature_range: tuple of (min, max) for scaled features
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_norm = df.copy()
    min_val, max_val = feature_range
    
    for col in columns:
        col_min = df_norm[col].min()
        col_max = df_norm[col].max()
        
        if col_max - col_min == 0:
            df_norm[col] = min_val
        else:
            df_norm[col] = min_val + (df_norm[col] - col_min) * (max_val - min_val) / (col_max - col_min)
    
    return df_norm

def standardize_zscore(df, columns=None):
    """
    Standardize data using z-score normalization.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to standardize (default: all numeric columns)
    
    Returns:
        DataFrame with standardized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_std = df.copy()
    
    for col in columns:
        mean_val = df_std[col].mean()
        std_val = df_std[col].std()
        
        if std_val == 0:
            df_std[col] = 0
        else:
            df_std[col] = (df_std[col] - mean_val) / std_val
    
    return df_std

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        df: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'drop'
        columns: list of column names to process (default: all columns)
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = df.columns
    
    df_handled = df.copy()
    
    for col in columns:
        if strategy == 'drop':
            df_handled = df_handled.dropna(subset=[col])
        elif strategy == 'mean' and pd.api.types.is_numeric_dtype(df_handled[col]):
            df_handled[col] = df_handled[col].fillna(df_handled[col].mean())
        elif strategy == 'median' and pd.api.types.is_numeric_dtype(df_handled[col]):
            df_handled[col] = df_handled[col].fillna(df_handled[col].median())
        elif strategy == 'mode':
            df_handled[col] = df_handled[col].fillna(df_handled[col].mode()[0])
    
    return df_handled

def create_data_pipeline(df, steps):
    """
    Create a data cleaning pipeline with multiple steps.
    
    Args:
        df: pandas DataFrame
        steps: list of tuples (function_name, kwargs)
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for step_func, kwargs in steps:
        if step_func == 'remove_outliers':
            cleaned_df = remove_outliers_iqr(cleaned_df, **kwargs)
        elif step_func == 'normalize':
            cleaned_df = normalize_minmax(cleaned_df, **kwargs)
        elif step_func == 'standardize':
            cleaned_df = standardize_zscore(cleaned_df, **kwargs)
        elif step_func == 'handle_missing':
            cleaned_df = handle_missing_values(cleaned_df, **kwargs)
    
    return cleaned_df