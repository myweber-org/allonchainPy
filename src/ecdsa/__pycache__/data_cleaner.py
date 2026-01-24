
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None, remove_duplicates=True, case_normalization='lower'):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    columns_to_clean (list, optional): List of column names to apply string normalization.
                                       If None, all object dtype columns are cleaned.
    remove_duplicates (bool): If True, remove duplicate rows.
    case_normalization (str): One of 'lower', 'upper', or None for case normalization.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if remove_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate row(s).")
    
    if columns_to_clean is None:
        columns_to_clean = cleaned_df.select_dtypes(include=['object']).columns.tolist()
    
    for col in columns_to_clean:
        if col in cleaned_df.columns and cleaned_df[col].dtype == 'object':
            cleaned_df[col] = cleaned_df[col].astype(str)
            
            cleaned_df[col] = cleaned_df[col].apply(lambda x: re.sub(r'\s+', ' ', x.strip()))
            
            if case_normalization == 'lower':
                cleaned_df[col] = cleaned_df[col].str.lower()
            elif case_normalization == 'upper':
                cleaned_df[col] = cleaned_df[col].str.upper()
    
    return cleaned_df

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    email_column (str): Name of the column containing email addresses.
    
    Returns:
    pd.DataFrame: DataFrame with additional 'email_valid' boolean column.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame.")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['email_valid'] = df[email_column].str.match(email_pattern, na=False)
    
    valid_count = df['email_valid'].sum()
    print(f"Found {valid_count} valid email addresses out of {len(df)} rows.")
    
    return df

if __name__ == "__main__":
    sample_data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', 'ALICE WONDER', ' bob '],
        'email': ['john@example.com', 'jane@test.org', 'invalid-email', 'alice@company.co.uk', 'bob@domain.com'],
        'age': [25, 30, 25, 28, 35]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned = clean_dataframe(df, case_normalization='lower')
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n")
    
    validated = validate_email_column(cleaned, 'email')
    print("DataFrame with email validation:")
    print(validated)
import pandas as pd
import numpy as np
from typing import Optional

def clean_csv_data(
    filepath: str,
    missing_strategy: str = 'drop',
    fill_value: Optional[float] = None,
    remove_duplicates: bool = True
) -> pd.DataFrame:
    """
    Load and clean CSV data with configurable missing value handling.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    missing_strategy : str
        Strategy for handling missing values: 'drop', 'fill', or 'ignore'
    fill_value : float, optional
        Value to fill missing entries when strategy is 'fill'
    remove_duplicates : bool
        Whether to remove duplicate rows
    
    Returns:
    --------
    pd.DataFrame
        Cleaned DataFrame
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    
    original_shape = df.shape
    
    if remove_duplicates:
        df = df.drop_duplicates()
    
    if missing_strategy == 'drop':
        df = df.dropna()
    elif missing_strategy == 'fill':
        if fill_value is not None:
            df = df.fillna(fill_value)
        else:
            df = df.fillna(df.mean(numeric_only=True))
    elif missing_strategy == 'ignore':
        pass
    else:
        raise ValueError("Invalid missing_strategy. Use 'drop', 'fill', or 'ignore'")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        df[numeric_cols] = df[numeric_cols].apply(
            lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99))
        )
    
    cleaned_shape = df.shape
    print(f"Data cleaning complete:")
    print(f"  Original shape: {original_shape}")
    print(f"  Cleaned shape: {cleaned_shape}")
    print(f"  Rows removed: {original_shape[0] - cleaned_shape[0]}")
    print(f"  Columns: {cleaned_shape[1]}")
    
    return df

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate DataFrame for common data quality issues.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate
    
    Returns:
    --------
    bool
        True if data passes validation checks
    """
    if df.empty:
        print("Validation failed: DataFrame is empty")
        return False
    
    if df.isnull().any().any():
        print("Validation warning: DataFrame contains missing values")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].std() == 0:
            print(f"Validation warning: Column '{col}' has zero variance")
    
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5, 5],
        'B': [10, 20, 30, np.nan, 50, 50],
        'C': ['x', 'y', 'z', 'x', 'y', 'y']
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data(
        'sample_data.csv',
        missing_strategy='fill',
        remove_duplicates=True
    )
    
    is_valid = validate_dataframe(cleaned_df)
    print(f"Data validation passed: {is_valid}")import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to clean.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
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

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing statistical measures.
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

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100)
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    print("Original statistics:", calculate_basic_stats(df, 'value'))
    
    cleaned_df = remove_outliers_iqr(df, 'value')
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned statistics:", calculate_basic_stats(cleaned_df, 'value'))
    
    return cleaned_df

if __name__ == "__main__":
    result_df = example_usage()
    print("\nFirst 5 rows of cleaned data:")
    print(result_df.head())