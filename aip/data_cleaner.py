
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    df_clean = df.copy()
    
    if drop_duplicates:
        initial_rows = df_clean.shape[0]
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - df_clean.shape[0]
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns
        
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                print(f"Filled missing values in numeric column '{col}' with median.")
        
        for col in categorical_cols:
            if df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna('Unknown')
                print(f"Filled missing values in categorical column '{col}' with 'Unknown'.")
    
    return df_clean

def validate_data(df, required_columns=None):
    """
    Validate the DataFrame for required columns and basic integrity.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        print("Warning: DataFrame is empty.")
    
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', None, 'Eve', 'Frank'],
        'age': [25, 30, 30, 35, None, 40],
        'score': [85.5, 92.0, 92.0, 78.5, 88.0, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    try:
        validate_data(cleaned_df, required_columns=['id', 'name', 'age'])
        print("\nData validation passed.")
    except ValueError as e:
        print(f"\nData validation failed: {e}")