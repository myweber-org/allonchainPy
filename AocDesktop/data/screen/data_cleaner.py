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
    Load and clean CSV data by handling missing values and removing specified columns.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save cleaned CSV file
        missing_strategy: Strategy for handling missing values ('mean', 'median', 'drop', 'zero')
        columns_to_drop: List of column names to remove from dataset
    
    Returns:
        Cleaned pandas DataFrame
    """
    
    # Load data
    df = pd.read_csv(input_path)
    
    # Remove specified columns
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if missing_strategy == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif missing_strategy == 'median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif missing_strategy == 'zero':
        df[numeric_cols] = df[numeric_cols].fillna(0)
    elif missing_strategy == 'drop':
        df = df.dropna(subset=numeric_cols)
    
    # Fill non-numeric columns with mode
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    
    return df

def validate_dataframe(df: pd.DataFrame) -> dict:
    """
    Validate DataFrame for common data quality issues.
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(exclude=[np.number]).columns)
    }
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    cleaned_df = clean_csv_data(
        input_path='raw_data.csv',
        output_path='cleaned_data.csv',
        missing_strategy='mean',
        columns_to_drop=['id', 'unused_column']
    )
    
    validation = validate_dataframe(cleaned_df)
    print(f"Data cleaning completed. Validation results: {validation}")