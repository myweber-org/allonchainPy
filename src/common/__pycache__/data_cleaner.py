
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True, strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (bool): Whether to fill missing values.
    strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', or 'constant').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
        print(f"Removed {len(df) - len(cleaned_df)} duplicate rows.")
    
    if fill_missing and cleaned_df.isnull().sum().any():
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        categorical_cols = cleaned_df.select_dtypes(exclude=[np.number]).columns
        
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = cleaned_df[col].mean()
                elif strategy == 'median':
                    fill_value = cleaned_df[col].median()
                elif strategy == 'constant':
                    fill_value = 0
                else:
                    fill_value = cleaned_df[col].mean()
                
                cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                print(f"Filled missing values in '{col}' with {fill_value:.2f}")
        
        for col in categorical_cols:
            if cleaned_df[col].isnull().any():
                mode_value = cleaned_df[col].mode().iloc[0] if not cleaned_df[col].mode().empty else 'Unknown'
                cleaned_df[col] = cleaned_df[col].fillna(mode_value)
                print(f"Filled missing values in '{col}' with '{mode_value}'")
    
    print(f"Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if len(df) < min_rows:
        print(f"Validation failed: Dataset has only {len(df)} rows, minimum required is {min_rows}")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    if df.isnull().sum().sum() > 0:
        print(f"Warning: Dataset contains {df.isnull().sum().sum()} missing values")
    
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, np.nan, 40.1, 50.0],
        'category': ['A', 'B', 'B', 'C', np.nan, 'A'],
        'score': [85, 92, 92, 78, 88, np.nan]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    if validate_dataset(df, required_columns=['id', 'value'], min_rows=3):
        cleaned_df = clean_dataset(df, drop_duplicates=True, fill_missing=True, strategy='mean')
        print("\nCleaned dataset:")
        print(cleaned_df)