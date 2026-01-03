import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = cleaned_df.shape[0]
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - cleaned_df.shape[0]
        print(f"Removed {removed} duplicate rows.")
    
    if cleaned_df.isnull().sum().any():
        print("Handling missing values...")
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        
        if fill_missing == 'mean':
            for col in numeric_cols:
                if cleaned_df[col].isnull().any():
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
        elif fill_missing == 'median':
            for col in numeric_cols:
                if cleaned_df[col].isnull().any():
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        elif fill_missing == 'zero':
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(0)
    
    object_cols = cleaned_df.select_dtypes(include=['object']).columns
    for col in object_cols:
        if cleaned_df[col].isnull().any():
            cleaned_df[col] = cleaned_df[col].fillna('Unknown')
    
    print(f"Data cleaning complete. Final shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 3, None, 5],
        'B': [10, None, 10, 20, 30, 40],
        'C': ['x', 'y', 'x', None, 'z', 'w'],
        'D': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaning DataFrame...")
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    try:
        validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
        print("\nData validation passed.")
    except ValueError as e:
        print(f"\nData validation failed: {e}")