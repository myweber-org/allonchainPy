import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Strategy to fill missing values. 
                        Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            elif fill_missing == 'median':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            elif fill_missing == 'mode':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (is_valid, message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', 'z', 'z']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='median')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B'])
    print(f"\nValidation: {message}")import pandas as pd
import numpy as np

def clean_csv_data(filepath, output_path=None):
    """
    Load a CSV file, perform basic cleaning operations,
    and optionally save the cleaned data.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded data with shape: {df.shape}")
        
        # Remove duplicate rows
        initial_count = len(df)
        df.drop_duplicates(inplace=True)
        removed_duplicates = initial_count - len(df)
        if removed_duplicates > 0:
            print(f"Removed {removed_duplicates} duplicate rows.")
        
        # Handle missing values: fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"Filled missing values in '{col}' with median: {median_val}")
        
        # Drop rows where all non-numeric essential columns are null
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            essential_null_mask = df[non_numeric_cols].isnull().all(axis=1)
            df = df[~essential_null_mask]
            print(f"Removed {essential_null_mask.sum()} rows with all non-numeric data missing.")
        
        # Reset index after cleaning
        df.reset_index(drop=True, inplace=True)
        print(f"Final cleaned data shape: {df.shape}")
        
        # Save if output path provided
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate the structure and content of a DataFrame.
    """
    if df is None or df.empty:
        print("DataFrame is empty or None.")
        return False
    
    print(f"DataFrame validation:")
    print(f"  - Shape: {df.shape}")
    print(f"  - Columns: {list(df.columns)}")
    print(f"  - Missing values per column:")
    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing > 0:
            print(f"    * {col}: {missing} missing")
    
    if required_columns:
        missing_req = [col for col in required_columns if col not in df.columns]
        if missing_req:
            print(f"  - Missing required columns: {missing_req}")
            return False
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_df = clean_csv_data("sample_data.csv", "cleaned_sample_data.csv")
    if sample_df is not None:
        validate_dataframe(sample_df)