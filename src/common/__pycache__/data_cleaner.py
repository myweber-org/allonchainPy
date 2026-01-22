import pandas as pd

def clean_dataset(df):
    """
    Clean a pandas DataFrame by removing duplicate rows and
    filling missing numeric values with column mean.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    
    # Fill missing numeric values with column mean
    numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
    df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())
    
    return df_cleaned

def validate_data(df, required_columns):
    """
    Validate that DataFrame contains all required columns.
    """
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'x', 'z', 'w']
    })
    
    print("Original data:")
    print(sample_data)
    
    cleaned_data = clean_dataset(sample_data)
    print("\nCleaned data:")
    print(cleaned_data)
    
    try:
        validate_data(cleaned_data, ['A', 'B'])
        print("\nData validation passed")
    except ValueError as e:
        print(f"\nData validation failed: {e}")