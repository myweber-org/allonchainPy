
import pandas as pd
import numpy as np
from typing import Optional

def clean_csv_data(
    input_path: str,
    output_path: str,
    missing_strategy: str = 'mean',
    columns_to_drop: Optional[list] = None
) -> pd.DataFrame:
    """
    Clean CSV data by handling missing values and removing specified columns.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save cleaned CSV file
        missing_strategy: Strategy for handling missing values ('mean', 'median', 'drop', 'zero')
        columns_to_drop: List of column names to remove
    
    Returns:
        Cleaned DataFrame
    """
    
    df = pd.read_csv(input_path)
    
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop, errors='ignore')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if missing_strategy == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif missing_strategy == 'median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif missing_strategy == 'zero':
        df[numeric_cols] = df[numeric_cols].fillna(0)
    elif missing_strategy == 'drop':
        df = df.dropna(subset=numeric_cols)
    
    df.to_csv(output_path, index=False)
    
    print(f"Data cleaned successfully. Rows: {len(df)}, Columns: {len(df.columns)}")
    print(f"Cleaned data saved to: {output_path}")
    
    return df

def detect_outliers_iqr(df: pd.DataFrame, column: str, threshold: float = 1.5) -> pd.Series:
    """
    Detect outliers using IQR method.
    
    Args:
        df: Input DataFrame
        column: Column name to check for outliers
        threshold: IQR multiplier threshold
    
    Returns:
        Boolean Series indicating outliers
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    return outliers

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, np.nan, 30, 40, 50],
        'C': ['x', 'y', 'z', 'x', 'y']
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data(
        input_path='sample_data.csv',
        output_path='cleaned_data.csv',
        missing_strategy='mean',
        columns_to_drop=['C']
    )
    
    print("\nSample cleaned data:")
    print(cleaned_df.head())