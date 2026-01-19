import pandas as pd

def clean_dataset(df, column_names=None):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        column_names (list, optional): List of column names to normalize. 
                                      If None, all object dtype columns are normalized.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    initial_rows = cleaned_df.shape[0]
    cleaned_df = cleaned_df.drop_duplicates()
    removed_duplicates = initial_rows - cleaned_df.shape[0]
    
    # Normalize string columns
    if column_names is None:
        # Select all object dtype columns (typically strings)
        column_names = cleaned_df.select_dtypes(include=['object']).columns.tolist()
    
    for col in column_names:
        if col in cleaned_df.columns and cleaned_df[col].dtype == 'object':
            # Convert to string, strip whitespace, and convert to lowercase
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    # Print cleaning summary
    print(f"Cleaning complete:")
    print(f"  - Removed {removed_duplicates} duplicate rows")
    print(f"  - Normalized {len(column_names)} columns")
    print(f"  - Final dataset shape: {cleaned_df.shape}")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of required column names.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    # Check for NaN values
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        print(f"Warning: Found {nan_count} NaN values in the DataFrame")
    
    return True

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', '  BOB JOHNSON  '],
        'email': ['john@example.com', 'jane@example.com', 'john@example.com', 'bob@example.com'],
        'age': [25, 30, 25, 35],
        'city': ['New York', 'Los Angeles', 'New York', 'CHICAGO']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned_df = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned data
    validation_result = validate_dataframe(cleaned_df, ['name', 'email'])
    print(f"\nData validation passed: {validation_result}")