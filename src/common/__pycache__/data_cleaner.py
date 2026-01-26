
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None):
    """
    Clean a pandas DataFrame by removing duplicate rows and normalizing string columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_clean (list, optional): List of column names to apply string normalization.
                                          If None, all object dtype columns are cleaned.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with duplicates removed and strings normalized.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    
    # Determine which columns to normalize
    if columns_to_clean is None:
        columns_to_clean = df_cleaned.select_dtypes(include=['object']).columns.tolist()
    
    # Apply string normalization to specified columns
    for col in columns_to_clean:
        if col in df_cleaned.columns and df_cleaned[col].dtype == 'object':
            df_cleaned[col] = df_cleaned[col].apply(_normalize_string)
    
    return df_cleaned

def _normalize_string(value):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping special characters from the beginning and end.
    
    Args:
        value: Input value to normalize. If not a string, returns as-is.
    
    Returns:
        Normalized string or original value.
    """
    if not isinstance(value, str):
        return value
    
    # Convert to lowercase
    normalized = value.lower()
    
    # Remove leading/trailing whitespace
    normalized = normalized.strip()
    
    # Replace multiple spaces with single space
    normalized = re.sub(r'\s+', ' ', normalized)
    
    return normalized

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column using a simple regex pattern.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        email_column (str): Name of the column containing email addresses.
    
    Returns:
        pd.DataFrame: DataFrame with an additional 'email_valid' boolean column.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    df = df.copy()
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['email_valid'] = df[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notnull(x) else False
    )
    
    return df