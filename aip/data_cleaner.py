
import pandas as pd

def clean_dataset(df, columns_to_check=None, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    columns_to_check (list, optional): Columns to check for duplicates. Defaults to all columns.
    fill_missing (bool): Whether to fill missing values. Defaults to True.
    fill_value: Value to fill missing entries with. Defaults to 0.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicates
    if columns_to_check is None:
        cleaned_df = cleaned_df.drop_duplicates()
    else:
        cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check)
    
    # Handle missing values
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate that DataFrame meets basic requirements.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "Data validation passed"

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', None],
        'score': [85, 90, 90, None, 88],
        'age': [25, 30, 30, 35, 28]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nDataFrame info:")
    print(df.info())
    
    # Clean the data
    cleaned = clean_dataset(df, columns_to_check=['id', 'name'], fill_missing=True)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    # Validate the cleaned data
    is_valid, message = validate_data(cleaned, required_columns=['id', 'name', 'score'])
    print(f"\nValidation: {message}")