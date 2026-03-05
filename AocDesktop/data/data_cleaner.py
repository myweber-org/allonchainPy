
import pandas as pd
import numpy as np
from typing import List, Optional

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

def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop', fill_value: any = None) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: 'drop' to remove rows, 'fill' to fill values
        fill_value: Value to use when strategy is 'fill'
    
    Returns:
        DataFrame with handled missing values
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill':
        return df.fillna(fill_value)
    else:
        raise ValueError("Strategy must be 'drop' or 'fill'")

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of columns that must be present
    
    Returns:
        True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    return len(missing_columns) == 0

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names by converting to lowercase and replacing spaces.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with cleaned column names
    """
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df

def filter_outliers(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
    """
    Remove outliers using z-score method.
    
    Args:
        df: Input DataFrame
        column: Column to check for outliers
        threshold: Z-score threshold
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in df.columns:
        return df
    
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    return df[z_scores < threshold]