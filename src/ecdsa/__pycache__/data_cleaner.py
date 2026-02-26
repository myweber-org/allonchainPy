
import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    subset (list, optional): Columns to consider for duplicates
    keep (str, optional): Which duplicates to keep - 'first', 'last', or False
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    original_shape = df.shape
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    removed_count = original_shape[0] - cleaned_df.shape[0]
    
    print(f"Removed {removed_count} duplicate rows")
    print(f"Original shape: {original_shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    
    return cleaned_df

def clean_missing_values(df, strategy='drop', fill_value=None):
    """
    Handle missing values in DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): 'drop' to remove rows, 'fill' to fill values
    fill_value: Value to fill when strategy is 'fill'
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if df.empty:
        return df
    
    if strategy == 'drop':
        cleaned_df = df.dropna()
        removed_count = df.shape[0] - cleaned_df.shape[0]
        print(f"Removed {removed_count} rows with missing values")
    elif strategy == 'fill':
        cleaned_df = df.fillna(fill_value)
        filled_count = df.isna().sum().sum()
        print(f"Filled {filled_count} missing values")
    else:
        raise ValueError("Strategy must be 'drop' or 'fill'")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    bool: True if validation passes
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    print(f"DataFrame validation passed")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return True

def main():
    """
    Example usage of data cleaning functions.
    """
    sample_data = {
        'id': [1, 2, 3, 1, 2, 4],
        'name': ['Alice', 'Bob', 'Charlie', 'Alice', 'Bob', 'David'],
        'score': [85, 92, 78, 85, 92, 88],
        'age': [25, 30, None, 25, 30, 35]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    validate_dataframe(df)
    
    print("\nRemoving duplicates:")
    df_clean = remove_duplicates(df, subset=['id', 'name'])
    print(df_clean)
    
    print("\nCleaning missing values:")
    df_filled = clean_missing_values(df_clean, strategy='fill', fill_value=0)
    print(df_filled)

if __name__ == "__main__":
    main()