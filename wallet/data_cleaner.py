
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    original_shape = df.shape
    
    if drop_duplicates:
        df = df.drop_duplicates()
        print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows")
    
    if fill_missing:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                if fill_missing == 'mean':
                    fill_value = df[col].mean()
                elif fill_missing == 'median':
                    fill_value = df[col].median()
                elif fill_missing == 'zero':
                    fill_value = 0
                else:
                    fill_value = fill_missing
                
                df[col] = df[col].fillna(fill_value)
                print(f"Filled missing values in column '{col}' with {fill_value}")
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna('Unknown')
            print(f"Filled missing values in categorical column '{col}' with 'Unknown'")
    
    print(f"Dataset cleaned: {original_shape} -> {df.shape}")
    return df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the dataset meets basic requirements.
    """
    if len(df) < min_rows:
        raise ValueError(f"Dataset must have at least {min_rows} rows")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'value': [10.5, np.nan, 15.0, 20.0, np.nan],
        'category': ['A', 'B', None, 'A', 'C']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    try:
        validate_data(cleaned_df, required_columns=['id', 'value'], min_rows=3)
        print("\nData validation passed!")
    except ValueError as e:
        print(f"\nData validation failed: {e}")