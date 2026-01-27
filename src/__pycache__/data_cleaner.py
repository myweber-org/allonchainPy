import pandas as pd
import numpy as np

def clean_dataframe(df):
    """
    Clean a pandas DataFrame by removing duplicates, handling missing values,
    and standardizing string columns.
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    initial_rows = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates()
    removed_duplicates = initial_rows - len(cleaned_df)
    
    # Standardize string columns (trim whitespace and convert to lowercase)
    string_columns = cleaned_df.select_dtypes(include=['object']).columns
    for col in string_columns:
        cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
    
    # Replace empty strings with NaN
    cleaned_df = cleaned_df.replace(r'^\s*$', np.nan, regex=True)
    
    # Fill missing numeric values with column mean
    numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
    
    # Fill missing string values with 'unknown'
    for col in string_columns:
        cleaned_df[col] = cleaned_df[col].fillna('unknown')
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df, removed_duplicates

def validate_dataframe(df):
    """
    Validate DataFrame for common data quality issues.
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'string_columns': list(df.select_dtypes(include=['object']).columns)
    }
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'name': ['Alice', 'Bob', 'Alice', 'Charlie', '  david  ', None],
        'age': [25, 30, 25, None, 35, 40],
        'city': ['New York', 'London', 'new york', 'Paris', '  BERLIN  ', ''],
        'score': [85.5, 92.0, 85.5, 78.5, 88.0, 91.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation Results:")
    print(validate_dataframe(df))
    
    cleaned_df, duplicates_removed = clean_dataframe(df)
    print(f"\nRemoved {duplicates_removed} duplicate rows")
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nCleaned Validation Results:")
    print(validate_dataframe(cleaned_df))