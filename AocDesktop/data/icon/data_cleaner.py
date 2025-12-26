
import re
import json
from typing import Dict, Any, Optional, List

def sanitize_string(input_string: str) -> str:
    """Remove extra whitespace and normalize line endings."""
    if not isinstance(input_string, str):
        raise TypeError("Input must be a string")
    cleaned = re.sub(r'\s+', ' ', input_string.strip())
    return cleaned.replace('\r\n', '\n').replace('\r', '\n')

def validate_email(email: str) -> bool:
    """Check if the provided string is a valid email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def parse_json_safe(json_string: str) -> Optional[Dict[str, Any]]:
    """Safely parse a JSON string, returning None on failure."""
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        return None

def filter_list_by_prefix(items: List[str], prefix: str) -> List[str]:
    """Filter a list of strings, keeping only those starting with a prefix."""
    return [item for item in items if item.startswith(prefix)]

def calculate_checksum(data: str) -> str:
    """Calculate a simple checksum for a string."""
    if not data:
        return '0'
    hash_val = 0
    for char in data:
        hash_val = (hash_val * 31 + ord(char)) & 0xFFFFFFFF
    return format(hash_val, '08x')
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (bool): Whether to fill missing values.
    fill_value: Value to use for filling missing data.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    return True, "DataFrame is valid"

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to process.
    method (str): 'iqr' for interquartile range or 'zscore' for standard deviation.
    threshold (float): Threshold for outlier detection.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    data = df[column].copy()
    
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (data >= lower_bound) & (data <= upper_bound)
    
    elif method == 'zscore':
        mean = data.mean()
        std = data.std()
        z_scores = np.abs((data - mean) / std)
        mask = z_scores <= threshold
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return df[mask]

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to normalize.
    method (str): Normalization method ('minmax' or 'zscore').
    
    Returns:
    pd.DataFrame: DataFrame with normalized column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
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
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df_copy

def sample_dataframe(df, sample_size=1000, random_state=42):
    """
    Sample rows from DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    sample_size (int): Number of rows to sample.
    random_state (int): Random seed for reproducibility.
    
    Returns:
    pd.DataFrame: Sampled DataFrame.
    """
    if len(df) <= sample_size:
        return df
    
    return df.sample(n=sample_size, random_state=random_state)
import pandas as pd
import numpy as np
from typing import List, Optional

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape

    def remove_duplicates(self, subset: Optional[List[str]] = None) -> 'DataCleaner':
        self.df = self.df.drop_duplicates(subset=subset, keep='first')
        return self

    def normalize_column(self, column_name: str, method: str = 'minmax') -> 'DataCleaner':
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame")

        if method == 'minmax':
            col_min = self.df[column_name].min()
            col_max = self.df[column_name].max()
            if col_max != col_min:
                self.df[column_name] = (self.df[column_name] - col_min) / (col_max - col_min)
        elif method == 'zscore':
            col_mean = self.df[column_name].mean()
            col_std = self.df[column_name].std()
            if col_std > 0:
                self.df[column_name] = (self.df[column_name] - col_mean) / col_std
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return self

    def fill_missing(self, column_name: str, strategy: str = 'mean') -> 'DataCleaner':
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame")

        if strategy == 'mean':
            fill_value = self.df[column_name].mean()
        elif strategy == 'median':
            fill_value = self.df[column_name].median()
        elif strategy == 'mode':
            fill_value = self.df[column_name].mode().iloc[0] if not self.df[column_name].mode().empty else np.nan
        elif strategy == 'zero':
            fill_value = 0
        else:
            raise ValueError(f"Unknown fill strategy: {strategy}")

        self.df[column_name] = self.df[column_name].fillna(fill_value)
        return self

    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df.copy()

    def get_stats(self) -> dict:
        cleaned_shape = self.df.shape
        return {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': cleaned_shape[0],
            'cleaned_columns': cleaned_shape[1],
            'rows_removed': self.original_shape[0] - cleaned_shape[0]
        }

def clean_dataset(df: pd.DataFrame, 
                  duplicate_columns: Optional[List[str]] = None,
                  normalize_columns: Optional[List[str]] = None,
                  fill_columns: Optional[List[str]] = None) -> pd.DataFrame:
    cleaner = DataCleaner(df)
    
    if duplicate_columns:
        cleaner.remove_duplicates(duplicate_columns)
    
    if normalize_columns:
        for col in normalize_columns:
            cleaner.normalize_column(col)
    
    if fill_columns:
        for col in fill_columns:
            cleaner.fill_missing(col)
    
    return cleaner.get_cleaned_data()