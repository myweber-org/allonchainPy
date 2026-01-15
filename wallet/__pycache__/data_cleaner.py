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

def fill_missing_values(df, strategy='mean', columns=None):
    """
    Fill missing values in DataFrame.
    
    Args:
        df: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'constant'
        columns: list of columns to fill, None for all columns
    
    Returns:
        DataFrame with missing values filled
    """
    df_filled = df.copy()
    
    if columns is None:
        columns = df.columns
    
    for col in columns:
        if df[col].dtype in ['int64', 'float64']:
            if strategy == 'mean':
                df_filled[col] = df[col].fillna(df[col].mean())
            elif strategy == 'median':
                df_filled[col] = df[col].fillna(df[col].median())
            elif strategy == 'mode':
                df_filled[col] = df[col].fillna(df[col].mode()[0])
            elif strategy == 'constant':
                df_filled[col] = df[col].fillna(0)
        else:
            df_filled[col] = df[col].fillna(df[col].mode()[0])
    
    return df_filled

def remove_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Remove outliers from DataFrame using specified method.
    
    Args:
        df: pandas DataFrame
        columns: list of numeric columns to check for outliers
        method: 'iqr' for interquartile range, 'zscore' for standard deviation
        threshold: multiplier for IQR or cutoff for z-score
    
    Returns:
        DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    
    if method == 'iqr':
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    elif method == 'zscore':
        for col in columns:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df_clean = df_clean[z_scores < threshold]
    
    return df_clean.reset_index(drop=True)

def standardize_columns(df, columns=None):
    """
    Standardize numeric columns to have zero mean and unit variance.
    
    Args:
        df: pandas DataFrame
        columns: list of columns to standardize
    
    Returns:
        DataFrame with standardized columns
    """
    df_std = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            df_std[col] = (df[col] - mean) / std
    
    return df_std

def clean_dataset(df, remove_dup=True, fill_na=True, remove_out=True, standardize=True):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: pandas DataFrame
        remove_dup: whether to remove duplicates
        fill_na: whether to fill missing values
        remove_out: whether to remove outliers
        standardize: whether to standardize numeric columns
    
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if remove_dup:
        df_clean = remove_duplicates(df_clean)
    
    if fill_na:
        df_clean = fill_missing_values(df_clean)
    
    if remove_out:
        df_clean = remove_outliers(df_clean)
    
    if standardize:
        df_clean = standardize_columns(df_clean)
    
    return df_clean
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (list or array-like): The dataset.
    column (int or str): The column index or name to process.
    
    Returns:
    numpy.ndarray: Data with outliers removed.
    """
    if isinstance(data, list):
        data = np.array(data)
    
    if isinstance(column, str):
        try:
            column_index = data.dtype.names.index(column)
            column_data = data[column]
        except (AttributeError, ValueError):
            raise ValueError("Column name not found in structured array")
    else:
        column_data = data[:, column]
    
    q1 = np.percentile(column_data, 25)
    q3 = np.percentile(column_data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (column_data >= lower_bound) & (column_data <= upper_bound)
    return data[mask]

def calculate_basic_stats(data, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    data (numpy.ndarray): The dataset.
    column (int or str): The column index or name.
    
    Returns:
    dict: Dictionary containing mean, median, and standard deviation.
    """
    if isinstance(column, str):
        try:
            column_data = data[column]
        except (TypeError, ValueError):
            raise ValueError("Invalid column name")
    else:
        column_data = data[:, column]
    
    stats = {
        'mean': np.mean(column_data),
        'median': np.median(column_data),
        'std': np.std(column_data)
    }
    return stats

if __name__ == "__main__":
    sample_data = np.array([
        [1.2, 150],
        [2.5, 200],
        [3.1, 180],
        [100.0, 5000],
        [4.2, 210],
        [5.7, 190]
    ])
    
    print("Original data:")
    print(sample_data)
    
    cleaned_data = remove_outliers_iqr(sample_data, 0)
    print("\nCleaned data (after removing outliers from column 0):")
    print(cleaned_data)
    
    stats = calculate_basic_stats(cleaned_data, 0)
    print("\nStatistics for column 0:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")