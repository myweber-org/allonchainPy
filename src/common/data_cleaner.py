import re
import pandas as pd
from typing import Union, List, Optional

def validate_email(email: str) -> bool:
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def clean_phone_number(phone: str) -> Optional[str]:
    digits = re.sub(r'\D', '', phone)
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    elif len(digits) == 11 and digits.startswith('1'):
        return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
    return None

def remove_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.DataFrame:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [col.lower().replace(' ', '_').strip() for col in df.columns]
    return df

def fill_missing_with_median(df: pd.DataFrame, columns: Union[str, List[str]]) -> pd.DataFrame:
    if isinstance(columns, str):
        columns = [columns]
    for col in columns:
        if col in df.columns:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
    return df
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

def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop', fill_value: Optional[float] = None) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: 'drop' to remove rows, 'fill' to replace with value
        fill_value: Value to use when strategy is 'fill'
    
    Returns:
        DataFrame with handled missing values
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill':
        if fill_value is not None:
            return df.fillna(fill_value)
        else:
            return df.fillna(df.mean(numeric_only=True))
    else:
        raise ValueError("Strategy must be 'drop' or 'fill'")

def normalize_numeric_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Normalize specified numeric columns to 0-1 range.
    
    Args:
        df: Input DataFrame
        columns: List of column names to normalize
    
    Returns:
        DataFrame with normalized columns
    """
    result = df.copy()
    for col in columns:
        if col in result.columns and pd.api.types.is_numeric_dtype(result[col]):
            col_min = result[col].min()
            col_max = result[col].max()
            if col_max > col_min:
                result[col] = (result[col] - col_min) / (col_max - col_min)
    return result

def clean_dataset(df: pd.DataFrame, 
                  deduplicate: bool = True,
                  missing_strategy: str = 'drop',
                  normalize_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        deduplicate: Whether to remove duplicates
        missing_strategy: Strategy for handling missing values
        normalize_cols: Columns to normalize
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if deduplicate:
        cleaned_df = remove_duplicates(cleaned_df)
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    if normalize_cols:
        cleaned_df = normalize_numeric_columns(cleaned_df, normalize_cols)
    
    return cleaned_df
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def normalize_minmax(df, columns):
    df_norm = df.copy()
    for col in columns:
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
    return df_norm

def clean_dataset(file_path, numeric_columns):
    df = pd.read_csv(file_path)
    df_clean = remove_outliers_iqr(df, numeric_columns)
    df_normalized = normalize_minmax(df_clean, numeric_columns)
    return df_normalized

if __name__ == "__main__":
    cleaned_data = clean_dataset('sample_data.csv', ['age', 'income', 'score'])
    cleaned_data.to_csv('cleaned_data.csv', index=False)
    print(f"Data cleaning complete. Original shape: {pd.read_csv('sample_data.csv').shape}, Cleaned shape: {cleaned_data.shape}")