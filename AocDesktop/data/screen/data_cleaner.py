
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing=True):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    df_cleaned = df.copy()
    
    if drop_duplicates:
        initial_rows = df_cleaned.shape[0]
        df_cleaned = df_cleaned.drop_duplicates()
        removed_rows = initial_rows - df_cleaned.shape[0]
        print(f"Removed {removed_rows} duplicate rows.")
    
    if fill_missing:
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        categorical_cols = df_cleaned.select_dtypes(exclude=[np.number]).columns
        
        for col in numeric_cols:
            if df_cleaned[col].isnull().any():
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
                print(f"Filled missing values in numeric column '{col}' with median.")
        
        for col in categorical_cols:
            if df_cleaned[col].isnull().any():
                df_cleaned[col] = df_cleaned[col].fillna('Unknown')
                print(f"Filled missing values in categorical column '{col}' with 'Unknown'.")
    
    return df_cleaned

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if df.empty:
        raise ValueError("DataFrame is empty.")
    
    print(f"DataFrame validation passed. Shape: {df.shape}")
    return True

def main():
    """
    Example usage of the data cleaning functions.
    """
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', None, 'Eve', 'Frank'],
        'age': [25, 30, 30, 35, None, 40],
        'score': [85.5, 92.0, 92.0, 78.5, 88.0, 95.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataframe(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    try:
        validate_dataframe(cleaned_df, required_columns=['id', 'name', 'age', 'score'])
    except ValueError as e:
        print(f"Validation error: {e}")

if __name__ == "__main__":
    main()