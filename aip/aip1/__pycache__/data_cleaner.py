import pandas as pd

def clean_dataset(df, column_names):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing specified string columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        column_names (list): List of column names to normalize (convert to lowercase and strip whitespace).
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with duplicates removed and strings normalized.
    """
    # Create a copy to avoid modifying the original DataFrame
    cleaned_df = df.copy()
    
    # Normalize specified string columns
    for col in column_names:
        if col in cleaned_df.columns and cleaned_df[col].dtype == 'object':
            cleaned_df[col] = cleaned_df[col].astype(str).str.lower().str.strip()
    
    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after dropping duplicates
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column using a simple regex pattern.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        email_column (str): Name of the column containing email addresses.
    
    Returns:
        pd.DataFrame: DataFrame with an additional 'email_valid' boolean column.
    """
    import re
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    df_copy = df.copy()
    df_copy['email_valid'] = df_copy[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notnull(x) else False
    )
    
    return df_copy

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     data = {
#         'name': ['John Doe', 'Jane Smith', 'John Doe', '  ALICE  '],
#         'email': ['john@example.com', 'invalid-email', 'john@example.com', 'alice@test.org'],
#         'age': [25, 30, 25, 28]
#     }
#     
#     df = pd.DataFrame(data)
#     print("Original DataFrame:")
#     print(df)
#     
#     # Clean the data
#     cleaned = clean_dataset(df, ['name'])
#     print("\nCleaned DataFrame:")
#     print(cleaned)
#     
#     # Validate emails
#     validated = validate_email_column(cleaned, 'email')
#     print("\nDataFrame with email validation:")
#     print(validated)