
import pandas as pd

def clean_dataset(df, columns_to_normalize=None):
    """
    Remove duplicate rows and normalize specified column names to lowercase.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    columns_to_normalize (list, optional): List of column names to normalize.
        If None, all columns are normalized.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    initial_rows = cleaned_df.shape[0]
    cleaned_df = cleaned_df.drop_duplicates()
    removed_rows = initial_rows - cleaned_df.shape[0]
    
    # Normalize column names
    if columns_to_normalize is None:
        columns_to_normalize = cleaned_df.columns.tolist()
    
    rename_dict = {}
    for col in columns_to_normalize:
        if col in cleaned_df.columns:
            rename_dict[col] = col.lower().strip()
    
    cleaned_df = cleaned_df.rename(columns=rename_dict)
    
    print(f"Removed {removed_rows} duplicate rows.")
    print(f"Normalized {len(rename_dict)} column names.")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and required columns.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list, optional): List of required column names.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame.")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty.")
        return True
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'Name': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob'],
        'Age': [25, 30, 25, 35, 30],
        'City': ['New York', 'London', 'New York', 'Paris', 'London']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned = clean_dataset(df)
    print(cleaned)
    
    # Validation test
    print("\nValidation test:")
    print(f"DataFrame validation: {validate_dataframe(cleaned, ['name', 'age'])}")