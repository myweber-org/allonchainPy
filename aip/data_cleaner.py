import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (list, optional): Column names to consider for duplicates
        keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    if subset is not None:
        if not all(col in df.columns for col in subset):
            raise ValueError("All subset columns must exist in DataFrame")
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate row(s)")
    
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(df):
    """
    Basic validation of DataFrame structure.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
    
    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        print("Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("DataFrame is empty")
        return False
    
    return True

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by converting to appropriate dtype.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to clean
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns
    """
    if not validate_dataframe(df):
        return df
    
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', 'David', 'Eve'],
        'score': ['85', '90', '90', '78', '92', '92', '88']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nAfter removing duplicates:")
    cleaned_df = remove_duplicates(df, subset=['id', 'name'])
    print(cleaned_df)
    
    print("\nAfter cleaning numeric column:")
    cleaned_df = clean_numeric_columns(cleaned_df, ['score'])
    print(cleaned_df.dtypes)
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, columns=None):
    """
    Remove outliers from DataFrame using Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to process. If None, process all numeric columns.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def standardize_columns(df, columns=None):
    """
    Standardize specified columns to have zero mean and unit variance.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to standardize
    
    Returns:
    pd.DataFrame: DataFrame with standardized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_standardized = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        mean_val = df[col].mean()
        std_val = df[col].std()
        
        if std_val > 0:
            df_standardized[col] = (df[col] - mean_val) / std_val
    
    return df_standardized

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame using specified strategy.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Strategy to handle missing values ('mean', 'median', 'mode', 'drop')
    columns (list): List of column names to process
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    if columns is None:
        columns = df.columns.tolist()
    
    df_processed = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if strategy == 'drop':
            df_processed = df_processed.dropna(subset=[col])
        elif strategy == 'mean':
            df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
        elif strategy == 'median':
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        elif strategy == 'mode':
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
    
    return df_processed.reset_index(drop=True)import re

def clean_string(text):
    """
    Cleans and normalizes a string by:
    - Converting to lowercase.
    - Removing leading/trailing whitespace.
    - Replacing multiple spaces with a single space.
    - Removing non-alphanumeric characters (except basic punctuation).
    """
    if not isinstance(text, str):
        return text

    # Convert to lowercase and strip whitespace
    text = text.lower().strip()

    # Replace multiple spaces/newlines/tabs with a single space
    text = re.sub(r'\s+', ' ', text)

    # Remove non-alphanumeric characters, but keep basic punctuation
    # This keeps letters, numbers, spaces, and .,!?;:-
    text = re.sub(r'[^a-z0-9\s.,!?;:-]', '', text)

    return text

def normalize_whitespace(text):
    """
    Normalizes whitespace in a string by ensuring single spaces between words.
    """
    if not isinstance(text, str):
        return text
    return ' '.join(text.split())
import pandas as pd
import numpy as np
from typing import List, Optional

def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None, keep: str = 'first') -> pd.DataFrame:
    """
    Remove duplicate rows from a pandas DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    subset (list, optional): Column names to consider for identifying duplicates
    keep (str): Which duplicates to keep - 'first', 'last', or False
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    if subset is not None:
        invalid_columns = [col for col in subset if col not in df.columns]
        if invalid_columns:
            raise ValueError(f"Columns not found in DataFrame: {invalid_columns}")
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep, ignore_index=True)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate row(s)")
    
    return cleaned_df

def clean_numeric_outliers(df: pd.DataFrame, column: str, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """
    Clean outliers in numeric columns using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    method (str): 'iqr' for interquartile range or 'zscore' for standard deviation
    threshold (float): Threshold multiplier for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers handled
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric")
    
    df_clean = df.copy()
    
    if method == 'iqr':
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = df_clean[(df_clean[column] < lower_bound) | (df_clean[column] > upper_bound)]
        df_clean.loc[outliers.index, column] = np.nan
        
    elif method == 'zscore':
        mean = df_clean[column].mean()
        std = df_clean[column].std()
        z_scores = np.abs((df_clean[column] - mean) / std)
        
        outliers = df_clean[z_scores > threshold]
        df_clean.loc[outliers.index, column] = np.nan
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    if len(outliers) > 0:
        print(f"Handled {len(outliers)} outlier(s) in column '{column}'")
    
    return df_clean

def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> bool:
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    bool: True if validation passes
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print("DataFrame contains missing values:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"  {col}: {count} missing values")
    
    return True

def main():
    """Example usage of data cleaning functions."""
    data = {
        'id': [1, 2, 3, 4, 4, 5, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'David', 'Eve', 'Eve'],
        'score': [85, 92, 78, 200, 200, 65, 65],
        'age': [25, 30, 35, 40, 40, 45, 45]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned_df = remove_duplicates(df, subset=['id', 'name'])
    print("After removing duplicates:")
    print(cleaned_df)
    print()
    
    validated = validate_dataframe(cleaned_df, required_columns=['id', 'name', 'score'])
    print(f"DataFrame validation: {validated}")
    print()
    
    outlier_cleaned = clean_numeric_outliers(cleaned_df, 'score', method='iqr')
    print("After cleaning outliers in 'score' column:")
    print(outlier_cleaned)

if __name__ == "__main__":
    main()