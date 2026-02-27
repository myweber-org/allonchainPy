import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list, optional): Column labels to consider for duplicates.
        keep (str, optional): Which duplicates to keep.
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    return cleaned_df

def clean_numeric_column(df, column_name):
    """
    Clean a numeric column by removing non-numeric characters.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Name of column to clean.
    
    Returns:
        pd.DataFrame: DataFrame with cleaned column.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    return df

def validate_email_format(df, email_column):
    """
    Validate email format and flag invalid entries.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        email_column (str): Name of email column.
    
    Returns:
        pd.DataFrame: DataFrame with validation results.
    """
    import re
    
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['email_valid'] = df[email_column].apply(
        lambda x: bool(re.match(pattern, str(x))) if pd.notnull(x) else False
    )
    return df

def main():
    """
    Example usage of data cleaning functions.
    """
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'name': ['Alice', 'Bob', 'Alice', 'Charlie', 'David', 'David'],
        'email': ['alice@example.com', 'bob@test', 'alice@example.com', 'charlie@domain.org', 'david@mail.com', 'david@mail.com'],
        'age': ['25', '30', '25', '35', '40', '40']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    df_clean = remove_duplicates(df, subset=['id', 'name'])
    print("After removing duplicates:")
    print(df_clean)
    print()
    
    df_clean = clean_numeric_column(df_clean, 'age')
    print("After cleaning numeric column:")
    print(df_clean)
    print()
    
    df_clean = validate_email_format(df_clean, 'email')
    print("After email validation:")
    print(df_clean)

if __name__ == "__main__":
    main()