
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
        raise ValueError("missing_strategy must be 'drop', 'fill', or 'ignore'")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        df[numeric_cols] = df[numeric_cols].apply(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x
        )
    
    print(f"Original shape: {original_shape}")
    print(f"Cleaned shape: {df.shape}")
    print(f"Removed rows: {original_shape[0] - df.shape[0]}")
    
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
        True if DataFrame passes validation
    """
    if df.empty:
        print("Warning: DataFrame is empty")
        return False
    
    if df.isnull().any().any():
        print("Warning: DataFrame contains missing values")
        return False
    
    if df.duplicated().any():
        print("Warning: DataFrame contains duplicates")
        return False
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].std() == 0:
            print(f"Warning: Column '{col}' has zero variance")
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [5, 6, 7, np.nan, 9],
        'C': [10, 11, 12, 13, 10]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned = clean_csv_data(
        'test_data.csv',
        missing_strategy='fill',
        remove_duplicates=True
    )
    
    print("\nValidation result:", validate_dataframe(cleaned))
    print("\nCleaned DataFrame:")
    print(cleaned)