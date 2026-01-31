
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        column_mapping (dict, optional): Dictionary mapping old column names to new ones
        drop_duplicates (bool): Whether to remove duplicate rows
        normalize_text (bool): Whether to normalize text columns (strip, lower case)
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    
    if normalize_text:
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
    
    return cleaned_df

def remove_special_characters(text, keep_pattern=r'[a-zA-Z0-9\s]'):
    """
    Remove special characters from text, keeping only specified patterns.
    
    Args:
        text (str): Input text
        keep_pattern (str): Regex pattern of characters to keep
    
    Returns:
        str: Cleaned text
    """
    if pd.isna(text):
        return text
    
    text = str(text)
    return re.sub(f'[^{keep_pattern}]', '', text)

def validate_email(email):
    """
    Validate email format using regex.
    
    Args:
        email (str): Email address to validate
    
    Returns:
        bool: True if email is valid, False otherwise
    """
    if pd.isna(email):
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email).strip()))

def clean_numeric_column(series, fill_na=0):
    """
    Clean numeric column by converting to numeric type and filling NaN values.
    
    Args:
        series (pd.Series): Numeric column to clean
        fill_na: Value to fill NaN with
    
    Returns:
        pd.Series: Cleaned numeric series
    """
    cleaned_series = pd.to_numeric(series, errors='coerce')
    return cleaned_series.fillna(fill_na)

if __name__ == "__main__":
    sample_data = {
        'Name': ['John Doe', 'Jane Smith', 'John Doe', ' Bob Johnson '],
        'Email': ['john@example.com', 'jane@example', 'john@example.com', 'bob@company.co.uk'],
        'Age': ['25', '30', '25', 'forty'],
        'Phone': ['(123) 456-7890', '987-654-3210', '(123) 456-7890', '555 123 4567']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned = clean_dataframe(df, drop_duplicates=True, normalize_text=True)
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n")
    
    cleaned['Email_Valid'] = cleaned['Email'].apply(validate_email)
    cleaned['Age_Clean'] = clean_numeric_column(cleaned['Age'])
    print("DataFrame with validation and numeric cleaning:")
    print(cleaned[['Name', 'Email', 'Email_Valid', 'Age', 'Age_Clean']])