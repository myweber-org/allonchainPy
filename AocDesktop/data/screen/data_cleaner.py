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
import pandas as pd
import numpy as np

def clean_dataframe(df):
    """
    Clean a pandas DataFrame by removing duplicates and standardizing column names.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    
    # Standardize column names: lowercase and replace spaces with underscores
    df_cleaned.columns = df_cleaned.columns.str.lower().str.replace(' ', '_')
    
    # Reset index after cleaning
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    return df_cleaned

def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in numeric columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == 'median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif strategy == 'mode':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mode().iloc[0])
    
    return df

def validate_dataframe(df):
    """
    Perform basic validation on DataFrame.
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if df.isnull().all().any():
        raise ValueError("DataFrame contains columns with all missing values")
    
    return True

def main():
    # Example usage
    data = {
        'Name': ['Alice', 'Bob', 'Alice', 'Charlie', None],
        'Age': [25, 30, 25, 35, 40],
        'Salary': [50000, 60000, 50000, 70000, 80000],
        'Department': ['HR', 'IT', 'HR', 'Finance', 'IT']
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean the data
    df_cleaned = clean_dataframe(df)
    print("After cleaning:")
    print(df_cleaned)
    print("\n")
    
    # Handle missing values
    df_filled = handle_missing_values(df_cleaned, strategy='mean')
    print("After handling missing values:")
    print(df_filled)
    print("\n")
    
    # Validate
    try:
        validate_dataframe(df_filled)
        print("Data validation passed")
    except ValueError as e:
        print(f"Data validation failed: {e}")

if __name__ == "__main__":
    main()