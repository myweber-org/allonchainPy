
import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None, fill_missing='mean', remove_duplicates=True):
    """
    Clean a pandas DataFrame by handling missing values and removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    columns_to_check (list): List of columns to check for missing values, defaults to all columns
    fill_missing (str): Method to fill missing values - 'mean', 'median', 'mode', or 'drop'
    remove_duplicates (bool): Whether to remove duplicate rows
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    df_clean = df.copy()
    
    if columns_to_check is None:
        columns_to_check = df_clean.columns.tolist()
    
    if remove_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - len(df_clean)
        print(f"Removed {removed} duplicate rows")
    
    for col in columns_to_check:
        if col in df_clean.columns:
            missing_count = df_clean[col].isnull().sum()
            if missing_count > 0:
                print(f"Column '{col}' has {missing_count} missing values")
                
                if fill_missing == 'drop':
                    df_clean = df_clean.dropna(subset=[col])
                elif fill_missing == 'mean' and pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                elif fill_missing == 'median' and pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                elif fill_missing == 'mode':
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
                else:
                    df_clean[col] = df_clean[col].fillna(0)
    
    print(f"Data cleaning complete. Original shape: {df.shape}, Cleaned shape: {df_clean.shape}")
    return df_clean

def validate_data(df, required_columns=None, numeric_columns=None):
    """
    Validate the cleaned dataset for required columns and numeric data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of columns that must be present
    numeric_columns (list): List of columns that must be numeric
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    if numeric_columns:
        non_numeric = []
        for col in numeric_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric.append(col)
        
        if non_numeric:
            print(f"Non-numeric columns found: {non_numeric}")
            return False
    
    if df.isnull().sum().sum() > 0:
        print("Dataset still contains missing values")
        return False
    
    print("Data validation passed")
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5, 6],
        'value': [10.5, np.nan, 15.2, 20.1, 25.0, 25.0, np.nan],
        'category': ['A', 'B', 'A', 'C', 'B', 'B', 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df, fill_missing='mean', remove_duplicates=True)
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    is_valid = validate_data(cleaned_df, required_columns=['id', 'value'], numeric_columns=['id', 'value'])
    print(f"\nDataset validation result: {is_valid}")