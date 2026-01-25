import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        df: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'drop'
        columns: specific columns to apply strategy to
    
    Returns:
        DataFrame with handled missing values
    """
    df_copy = df.copy()
    
    if columns is None:
        columns = df_copy.columns
    
    for col in columns:
        if df_copy[col].dtype in ['int64', 'float64']:
            if strategy == 'mean':
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)
            elif strategy == 'median':
                df_copy[col].fillna(df_copy[col].median(), inplace=True)
            elif strategy == 'mode':
                df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                df_copy = df_copy.dropna(subset=[col])
    
    return df_copy

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column in DataFrame.
    
    Args:
        df: pandas DataFrame
        column: column name to normalize
        method: 'minmax' or 'zscore'
    
    Returns:
        DataFrame with normalized column
    """
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        if max_val != min_val:
            df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val != 0:
            df_copy[column] = (df_copy[column] - mean_val) / std_val
    
    return df_copy

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a column in DataFrame.
    
    Args:
        df: pandas DataFrame
        column: column name to check for outliers
        method: 'iqr' or 'zscore'
        threshold: threshold value for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    df_copy = df.copy()
    
    if method == 'iqr':
        Q1 = df_copy[column].quantile(0.25)
        Q3 = df_copy[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        df_copy = df_copy[(df_copy[column] >= lower_bound) & (df_copy[column] <= upper_bound)]
    
    elif method == 'zscore':
        z_scores = np.abs((df_copy[column] - df_copy[column].mean()) / df_copy[column].std())
        df_copy = df_copy[z_scores < threshold]
    
    return df_copy

def clean_data_pipeline(df, steps=None):
    """
    Execute a pipeline of data cleaning steps.
    
    Args:
        df: pandas DataFrame
        steps: list of cleaning functions and their arguments
    
    Returns:
        Cleaned DataFrame
    """
    if steps is None:
        steps = [
            (remove_duplicates, {}),
            (handle_missing_values, {'strategy': 'mean'}),
            (remove_outliers, {'column': df.columns[0], 'method': 'iqr'})
        ]
    
    cleaned_df = df.copy()
    
    for func, kwargs in steps:
        cleaned_df = func(cleaned_df, **kwargs)
    
    return cleaned_dfimport numpy as np
import pandas as pd
from scipy import stats

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
        return df[column].apply(lambda x: 0.5)
    return (df[column] - min_val) / (max_val - min_val)

def standardize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    if std_val == 0:
        return df[column].apply(lambda x: 0)
    return (df[column] - mean_val) / std_val

def clean_dataset(df, numeric_columns, outlier_threshold=1.5):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

def process_features(df, features, method='standardize'):
    processed_df = df.copy()
    for feature in features:
        if feature in processed_df.columns:
            if method == 'normalize':
                processed_df[feature] = normalize_minmax(processed_df, feature)
            elif method == 'standardize':
                processed_df[feature] = standardize_zscore(processed_df, feature)
    return processed_df

def validate_data(df, required_columns):
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    return True