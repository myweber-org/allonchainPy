import pandas as pd

def clean_dataframe(df):
    """
    Remove rows with null values and standardize column names.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Remove rows with any null values
    df_cleaned = df.dropna()
    
    # Standardize column names: lowercase and replace spaces with underscores
    df_cleaned.columns = (
        df_cleaned.columns
        .str.lower()
        .str.replace(' ', '_')
        .str.replace(r'[^a-z0-9_]', '', regex=True)
    )
    
    return df_cleaned

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
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
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'Product Name': ['Widget A', 'Widget B', None, 'Widget C'],
        'Price': [10.99, 15.49, 20.99, 12.99],
        'In Stock': [True, False, True, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned_df = clean_dataframe(df)
    print(cleaned_df)
    
    is_valid, message = validate_dataframe(cleaned_df, ['product_name', 'price'])
    print(f"\nValidation: {is_valid} - {message}")