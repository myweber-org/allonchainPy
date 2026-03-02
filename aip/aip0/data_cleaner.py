import pandas as pd

def clean_dataset(df, columns_to_check=None, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_check (list, optional): Specific columns to check for duplicates.
                                          If None, checks all columns.
        fill_missing (str or value): Method to fill missing values.
                                    'mean', 'median', 'mode', or a specific value.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    original_shape = df.shape
    
    # Remove duplicates
    if columns_to_check:
        df_clean = df.drop_duplicates(subset=columns_to_check)
    else:
        df_clean = df.drop_duplicates()
    
    # Handle missing values
    if fill_missing == 'mean':
        df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))
    elif fill_missing == 'median':
        df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
    elif fill_missing == 'mode':
        df_clean = df_clean.fillna(df_clean.mode().iloc[0])
    else:
        df_clean = df_clean.fillna(fill_missing)
    
    # Report cleaning statistics
    duplicates_removed = original_shape[0] - df_clean.shape[0]
    missing_filled = df.isna().sum().sum() - df_clean.isna().sum().sum()
    
    print(f"Original dataset shape: {original_shape}")
    print(f"Cleaned dataset shape: {df_clean.shape}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Missing values filled: {missing_filled}")
    
    return df_clean

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
        min_rows (int): Minimum number of rows required.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Error: DataFrame is empty")
        return False
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    return True

# Example usage
if __name__ == "__main__":
    # Create sample data with duplicates and missing values
    data = {
        'id': [1, 2, 3, 1, 2, 6],
        'name': ['Alice', 'Bob', 'Charlie', 'Alice', 'Bob', 'David'],
        'age': [25, 30, None, 25, 30, 35],
        'score': [85.5, 92.0, 78.5, 85.5, 92.0, None]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the dataset
    cleaned_df = clean_dataset(df, columns_to_check=['id', 'name'], fill_missing='mean')
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned DataFrame
    is_valid = validate_dataframe(cleaned_df, required_columns=['id', 'name', 'age', 'score'])
    print(f"\nDataFrame validation: {'PASSED' if is_valid else 'FAILED'}")