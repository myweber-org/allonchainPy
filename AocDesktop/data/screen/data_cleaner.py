
def remove_duplicates_preserve_order(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport pandas as pd
import numpy as np
from typing import Optional, List

def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df: pd.DataFrame, strategy: str = 'mean', columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Fill missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: 'mean', 'median', 'mode', or 'constant'
        columns: Specific columns to fill
    
    Returns:
        DataFrame with filled missing values
    """
    df_filled = df.copy()
    
    if columns is None:
        columns = df.columns
    
    for col in columns:
        if df[col].isnull().any():
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0]
            elif strategy == 'constant':
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            df_filled[col] = df[col].fillna(fill_value)
    
    return df_filled

def normalize_column(df: pd.DataFrame, column: str, method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize a column in DataFrame.
    
    Args:
        df: Input DataFrame
        column: Column to normalize
        method: 'minmax' or 'zscore'
    
    Returns:
        DataFrame with normalized column
    """
    df_normalized = df.copy()
    
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val != min_val:
            df_normalized[column] = (df[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val > 0:
            df_normalized[column] = (df[column] - mean_val) / std_val
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return df_normalized

def detect_outliers_iqr(df: pd.DataFrame, column: str, threshold: float = 1.5) -> pd.Series:
    """
    Detect outliers using IQR method.
    
    Args:
        df: Input DataFrame
        column: Column to check for outliers
        threshold: IQR multiplier threshold
    
    Returns:
        Boolean Series indicating outliers
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def clean_data_pipeline(df: pd.DataFrame, 
                       remove_dup_cols: Optional[List[str]] = None,
                       fill_strategy: str = 'mean',
                       normalize_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Complete data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        remove_dup_cols: Columns for duplicate removal
        fill_strategy: Strategy for filling missing values
        normalize_cols: Columns to normalize
    
    Returns:
        Cleaned DataFrame
    """
    # Step 1: Remove duplicates
    df_clean = remove_duplicates(df, remove_dup_cols)
    
    # Step 2: Fill missing values
    df_clean = fill_missing_values(df_clean, strategy=fill_strategy)
    
    # Step 3: Normalize specified columns
    if normalize_cols:
        for col in normalize_cols:
            if col in df_clean.columns:
                df_clean = normalize_column(df_clean, col, method='minmax')
    
    return df_clean
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
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

def clean_dataset(df, columns_to_clean):
    """
    Clean multiple columns in a dataset by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary of summary statistics for each cleaned column
    """
    cleaned_df = df.copy()
    stats_dict = {}
    
    for column in columns_to_clean:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            
            stats = calculate_summary_stats(cleaned_df, column)
            stats['outliers_removed'] = removed_count
            stats_dict[column] = stats
    
    return cleaned_df, stats_dict