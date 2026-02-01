
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    original_shape = df.shape
    
    if drop_duplicates:
        df = df.drop_duplicates()
        print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows.")
    
    if df.isnull().sum().any():
        missing_counts = df.isnull().sum()
        print("Missing values per column:")
        print(missing_counts[missing_counts > 0])
        
        if fill_missing == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            print("Filled numeric missing values with column mean.")
        elif fill_missing == 'median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            print("Filled numeric missing values with column median.")
        elif fill_missing == 'mode':
            for col in df.columns:
                if df[col].isnull().any():
                    mode_val = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_val)
            print("Filled missing values with column mode.")
        elif fill_missing == 'drop':
            df = df.dropna()
            print("Dropped rows with missing values.")
    
    print(f"Dataset cleaned. Original shape: {original_shape}, New shape: {df.shape}")
    return df

def validate_dataset(df, required_columns=None, unique_constraints=None):
    """
    Validate dataset structure and constraints.
    """
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if unique_constraints:
        for col in unique_constraints:
            if df[col].duplicated().any():
                duplicates = df[df[col].duplicated(keep=False)]
                print(f"Warning: Column '{col}' has duplicate values.")
                print(f"Duplicate count: {len(duplicates)}")
    
    print("Dataset validation completed.")
    return True

if __name__ == "__main__":
    # Example usage
    data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10, 20, 20, np.nan, 40, 50],
        'category': ['A', 'B', 'B', 'A', None, 'C']
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    validation_passed = validate_dataset(
        cleaned_df, 
        required_columns=['id', 'value'],
        unique_constraints=['id']
    )