
import pandas as pd
import numpy as np
from typing import List, Optional

def remove_duplicate_rows(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def normalize_string_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Normalize string columns by stripping whitespace and converting to lowercase.
    
    Args:
        df: Input DataFrame
        columns: List of column names to normalize
    
    Returns:
        DataFrame with normalized string columns
    """
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns and df_copy[col].dtype == 'object':
            df_copy[col] = df_copy[col].astype(str).str.strip().str.lower()
    return df_copy

def clean_numeric_outliers(df: pd.DataFrame, columns: List[str], 
                          method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """
    Clean numeric outliers using specified method.
    
    Args:
        df: Input DataFrame
        columns: Numeric columns to clean
        method: 'iqr' for interquartile range or 'zscore' for standard deviation
        threshold: Threshold for outlier detection
    
    Returns:
        DataFrame with outliers replaced by NaN
    """
    df_copy = df.copy()
    
    for col in columns:
        if col not in df_copy.columns or not np.issubdtype(df_copy[col].dtype, np.number):
            continue
            
        if method == 'iqr':
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask = (df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((df_copy[col] - df_copy[col].mean()) / df_copy[col].std())
            mask = z_scores > threshold
            
        else:
            continue
            
        df_copy.loc[mask, col] = np.nan
    
    return df_copy

def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop', 
                         fill_value: Optional[float] = None) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: 'drop' to remove rows, 'fill' to fill values
        fill_value: Value to use when filling (only for numeric columns if None)
    
    Returns:
        DataFrame with handled missing values
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill':
        if fill_value is not None:
            return df.fillna(fill_value)
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df_copy = df.copy()
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(df_copy[numeric_cols].mean())
            return df_copy
    return df

def clean_dataframe(df: pd.DataFrame, 
                   duplicate_subset: Optional[List[str]] = None,
                   string_columns: Optional[List[str]] = None,
                   numeric_columns: Optional[List[str]] = None,
                   missing_strategy: str = 'drop') -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        duplicate_subset: Columns for duplicate removal
        string_columns: String columns to normalize
        numeric_columns: Numeric columns for outlier cleaning
        missing_strategy: Strategy for handling missing values
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    cleaned_df = remove_duplicate_rows(cleaned_df, duplicate_subset)
    
    if string_columns:
        cleaned_df = normalize_string_columns(cleaned_df, string_columns)
    
    if numeric_columns:
        cleaned_df = clean_numeric_outliers(cleaned_df, numeric_columns)
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    return cleaned_df
import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None, missing_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Args:
        file_path (str): Path to input CSV file
        output_path (str, optional): Path for cleaned output CSV. 
                                    If None, returns DataFrame
        missing_strategy (str): Strategy for handling missing values.
                               Options: 'mean', 'median', 'drop', 'zero'
    
    Returns:
        pd.DataFrame or None: Cleaned DataFrame if output_path is None
    """
    
    try:
        df = pd.read_csv(file_path)
        
        # Remove duplicate rows
        initial_rows = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if missing_strategy == 'mean':
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].mean())
        elif missing_strategy == 'median':
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].median())
        elif missing_strategy == 'zero':
            df[numeric_cols] = df[numeric_cols].fillna(0)
        elif missing_strategy == 'drop':
            df = df.dropna(subset=numeric_cols)
        
        # Reset index after cleaning
        df = df.reset_index(drop=True)
        
        # Log cleaning summary
        print(f"Data cleaning completed:")
        print(f"  - Removed {duplicates_removed} duplicate rows")
        print(f"  - Final dataset: {len(df)} rows, {len(df.columns)} columns")
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"  - Cleaned data saved to: {output_path}")
            return None
        else:
            return df
            
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        bool: True if validation passes
    """
    if df is None or df.empty:
        print("Validation failed: DataFrame is empty or None")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing columns {missing_cols}")
            return False
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    
    if inf_count > 0:
        print(f"Warning: Found {inf_count} infinite values")
    
    return True

if __name__ == "__main__":
    # Example usage
    cleaned_data = clean_csv_data(
        file_path='raw_data.csv',
        output_path='cleaned_data.csv',
        missing_strategy='mean'
    )
    
    if cleaned_data is not None:
        is_valid = validate_dataframe(cleaned_data)
        print(f"Data validation: {'Passed' if is_valid else 'Failed'}")