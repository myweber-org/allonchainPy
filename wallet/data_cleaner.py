import pandas as pd
import numpy as np
from typing import Optional

def clean_csv_data(
    input_path: str,
    output_path: str,
    missing_strategy: str = 'mean',
    numeric_columns: Optional[list] = None
) -> pd.DataFrame:
    """
    Clean CSV data by handling missing values and converting data types.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save cleaned CSV file
        missing_strategy: Strategy for handling missing values ('mean', 'median', 'drop', 'zero')
        numeric_columns: List of column names to treat as numeric
    
    Returns:
        Cleaned DataFrame
    """
    
    # Read the CSV file
    df = pd.read_csv(input_path)
    
    # Identify numeric columns if not specified
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Handle missing values based on strategy
    if missing_strategy == 'mean':
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())
    elif missing_strategy == 'median':
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
    elif missing_strategy == 'zero':
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0)
    elif missing_strategy == 'drop':
        df = df.dropna(subset=numeric_columns)
    
    # Convert numeric columns to appropriate types
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Reset index after cleaning
    df = df.reset_index(drop=True)
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    
    return df

def validate_dataframe(df: pd.DataFrame) -> dict:
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'row_count': len(df),
        'column_count': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.astype(str).to_dict()
    }
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    cleaned_df = clean_csv_data(
        input_path='raw_data.csv',
        output_path='cleaned_data.csv',
        missing_strategy='mean'
    )
    
    validation = validate_dataframe(cleaned_df)
    print(f"Data cleaning completed. Validation results: {validation}")