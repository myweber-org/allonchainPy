
import pandas as pd
import numpy as np
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
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
    return cleaned_df

def calculate_statistics(df):
    stats_dict = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        stats_dict[col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'skewness': stats.skew(df[col].dropna()),
            'kurtosis': stats.kurtosis(df[col].dropna())
        }
    return stats_dict

def process_dataframe(input_df):
    numeric_cols = input_df.select_dtypes(include=[np.number]).columns.tolist()
    cleaned_df = clean_dataset(input_df, numeric_cols)
    statistics = calculate_statistics(cleaned_df)
    return cleaned_df, statistics
import pandas as pd

def clean_dataframe(df, drop_na=True, rename_columns=True):
    """
    Clean a pandas DataFrame by removing null values and standardizing column names.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_na (bool): If True, drop rows with any null values.
    rename_columns (bool): If True, rename columns to lowercase with underscores.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_na:
        cleaned_df = cleaned_df.dropna()
    
    if rename_columns:
        cleaned_df.columns = (
            cleaned_df.columns
            .str.lower()
            .str.replace(' ', '_')
            .str.replace(r'[^a-z0-9_]', '', regex=True)
        )
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and required columns.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'User ID': [1, 2, None, 4],
        'First Name': ['Alice', 'Bob', 'Charlie', None],
        'Last Name': ['Smith', 'Johnson', 'Brown', 'Davis'],
        'Age': [25, 30, 35, 40]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned = clean_dataframe(df)
    print(cleaned)
    
    is_valid = validate_dataframe(cleaned, ['user_id', 'first_name', 'last_name', 'age'])
    print(f"\nDataFrame validation: {is_valid}")
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

def normalize_column(df: pd.DataFrame, column: str, method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize specified column using different methods.
    
    Args:
        df: Input DataFrame
        column: Column name to normalize
        method: Normalization method ('minmax', 'zscore', 'log')
    
    Returns:
        DataFrame with normalized column
    """
    df = df.copy()
    
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val > min_val:
            df[column] = (df[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val > 0:
            df[column] = (df[column] - mean_val) / std_val
    
    elif method == 'log':
        if df[column].min() > 0:
            df[column] = np.log(df[column])
    
    return df

def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Handle missing values in numeric columns.
    
    Args:
        df: Input DataFrame
        strategy: Imputation strategy ('mean', 'median', 'mode', 'drop')
    
    Returns:
        DataFrame with handled missing values
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if df[col].isnull().any():
            if strategy == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == 'mode':
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                df = df.dropna(subset=[col])
    
    return df

def clean_dataframe(df: pd.DataFrame, 
                   remove_dups: bool = True,
                   normalize_cols: Optional[List[str]] = None,
                   missing_strategy: str = 'mean') -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        remove_dups: Whether to remove duplicates
        normalize_cols: Columns to normalize
        missing_strategy: Strategy for handling missing values
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if remove_dups:
        cleaned_df = remove_duplicates(cleaned_df)
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    if normalize_cols:
        for col in normalize_cols:
            if col in cleaned_df.columns:
                cleaned_df = normalize_column(cleaned_df, col)
    
    return cleaned_df