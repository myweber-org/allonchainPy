
import pandas as pd
import numpy as np
from typing import Union, List, Optional

def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: Input DataFrame.
        subset: List of column names to consider for identifying duplicates.
                If None, all columns are used.
    
    Returns:
        DataFrame with duplicates removed.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df: pd.DataFrame, 
                        strategy: str = 'mean', 
                        columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Fill missing values in specified columns using a given strategy.
    
    Args:
        df: Input DataFrame.
        strategy: Method to use for filling missing values.
                 Options: 'mean', 'median', 'mode', 'zero'.
        columns: List of column names to process. If None, all numeric columns are processed.
    
    Returns:
        DataFrame with missing values filled.
    """
    df_filled = df.copy()
    
    if columns is None:
        columns = df_filled.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col not in df_filled.columns:
            continue
            
        if strategy == 'mean':
            fill_value = df_filled[col].mean()
        elif strategy == 'median':
            fill_value = df_filled[col].median()
        elif strategy == 'mode':
            fill_value = df_filled[col].mode()[0] if not df_filled[col].mode().empty else 0
        elif strategy == 'zero':
            fill_value = 0
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")
        
        df_filled[col].fillna(fill_value, inplace=True)
    
    return df_filled

def validate_numeric_range(df: pd.DataFrame, 
                          column: str, 
                          min_val: Union[int, float] = -np.inf, 
                          max_val: Union[int, float] = np.inf) -> pd.Series:
    """
    Validate if values in a column are within specified range.
    
    Args:
        df: Input DataFrame.
        column: Column name to validate.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.
    
    Returns:
        Boolean Series indicating valid rows.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    return (df[column] >= min_val) & (df[column] <= max_val)

def normalize_column(df: pd.DataFrame, 
                    column: str, 
                    method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize values in a column using specified method.
    
    Args:
        df: Input DataFrame.
        column: Column name to normalize.
        method: Normalization method. Options: 'minmax', 'zscore'.
    
    Returns:
        DataFrame with normalized column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_normalized = df.copy()
    
    if method == 'minmax':
        col_min = df_normalized[column].min()
        col_max = df_normalized[column].max()
        if col_max != col_min:
            df_normalized[column] = (df_normalized[column] - col_min) / (col_max - col_min)
    
    elif method == 'zscore':
        col_mean = df_normalized[column].mean()
        col_std = df_normalized[column].std()
        if col_std > 0:
            df_normalized[column] = (df_normalized[column] - col_mean) / col_std
    
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    
    return df_normalized

def clean_dataset(df: pd.DataFrame, 
                 remove_dups: bool = True,
                 fill_na: bool = True,
                 fill_strategy: str = 'mean',
                 normalize_cols: Optional[List[str]] = None,
                 normalization_method: str = 'minmax') -> pd.DataFrame:
    """
    Apply a complete cleaning pipeline to a dataset.
    
    Args:
        df: Input DataFrame.
        remove_dups: Whether to remove duplicate rows.
        fill_na: Whether to fill missing values.
        fill_strategy: Strategy for filling missing values.
        normalize_cols: List of columns to normalize.
        normalization_method: Method for normalization.
    
    Returns:
        Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if remove_dups:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if fill_na:
        cleaned_df = fill_missing_values(cleaned_df, strategy=fill_strategy)
    
    if normalize_cols:
        for col in normalize_cols:
            if col in cleaned_df.columns:
                cleaned_df = normalize_column(cleaned_df, col, method=normalization_method)
    
    return cleaned_df
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

def calculate_summary_statistics(df, column):
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
        'count': len(df[column])
    }
    
    return stats

def clean_dataset(df, columns_to_clean):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary of statistics for each cleaned column
    """
    cleaned_df = df.copy()
    all_stats = {}
    
    for column in columns_to_clean:
        if column in cleaned_df.columns:
            original_stats = calculate_summary_statistics(cleaned_df, column)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            cleaned_stats = calculate_summary_statistics(cleaned_df, column)
            
            all_stats[column] = {
                'original': original_stats,
                'cleaned': cleaned_stats,
                'removed_count': len(df) - len(cleaned_df)
            }
    
    return cleaned_df, all_stats