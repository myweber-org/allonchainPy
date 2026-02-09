import pandas as pd
import numpy as np

def clean_dataframe(df):
    """
    Clean a pandas DataFrame by removing null values and standardizing column names.
    """
    # Remove rows with any null values
    df_cleaned = df.dropna()
    
    # Standardize column names: lowercase and replace spaces with underscores
    df_cleaned.columns = df_cleaned.columns.str.lower().str.replace(' ', '_')
    
    # Reset index after dropping rows
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    return df_cleaned

def validate_dataframe(df):
    """
    Validate that the DataFrame has no null values and column names are standardized.
    """
    # Check for null values
    if df.isnull().sum().sum() > 0:
        raise ValueError("DataFrame contains null values after cleaning.")
    
    # Check column name format
    for col in df.columns:
        if not col.islower() or ' ' in col:
            raise ValueError(f"Column name '{col}' is not properly standardized.")
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'First Name': ['Alice', 'Bob', None, 'David'],
        'Last Name': ['Smith', 'Johnson', 'Williams', None],
        'Age': [25, 30, 35, 40],
        'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned_df = clean_dataframe(df)
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n")
    
    try:
        validate_dataframe(cleaned_df)
        print("Data validation passed.")
    except ValueError as e:
        print(f"Data validation failed: {e}")