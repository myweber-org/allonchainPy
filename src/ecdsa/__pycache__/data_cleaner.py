
import pandas as pd

def remove_duplicates(dataframe, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        dataframe: pandas DataFrame to clean
        subset: column label or sequence of labels to consider for duplicates
        keep: determines which duplicates to keep ('first', 'last', or False)
    
    Returns:
        Cleaned DataFrame with duplicates removed
    """
    if dataframe.empty:
        return dataframe
    
    cleaned_df = dataframe.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(dataframe) - len(cleaned_df)
    print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_numeric_columns(dataframe, columns):
    """
    Clean numeric columns by removing non-numeric values.
    
    Args:
        dataframe: pandas DataFrame
        columns: list of column names to clean
    
    Returns:
        DataFrame with cleaned numeric columns
    """
    for col in columns:
        if col in dataframe.columns:
            dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
    
    return dataframe

def validate_dataframe(dataframe, required_columns):
    """
    Validate that DataFrame contains required columns.
    
    Args:
        dataframe: pandas DataFrame to validate
        required_columns: list of required column names
    
    Returns:
        Boolean indicating if validation passed
    """
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return False
    
    return True
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df: pandas DataFrame to clean
        column_mapping: Optional dictionary to rename columns
        drop_duplicates: Whether to remove duplicate rows
        normalize_text: Whether to normalize text columns (strip, lower case)
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if normalize_text:
        for column in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[column] = cleaned_df[column].apply(
                lambda x: str(x).strip().lower() if pd.notnull(x) else x
            )
    
    return cleaned_df

def validate_email(email):
    """
    Validate email format using regex.
    
    Args:
        email: Email string to validate
    
    Returns:
        Boolean indicating if email is valid
    """
    if pd.isnull(email):
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email).strip()))

def filter_valid_emails(df, email_column='email'):
    """
    Filter DataFrame to only include rows with valid email addresses.
    
    Args:
        df: pandas DataFrame
        email_column: Name of the column containing email addresses
    
    Returns:
        DataFrame with only valid email rows
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    mask = df[email_column].apply(validate_email)
    valid_df = df[mask].reset_index(drop=True)
    
    print(f"Found {len(valid_df)} rows with valid emails out of {len(df)} total rows")
    return valid_df

def save_cleaned_data(df, output_path, format='csv'):
    """
    Save cleaned DataFrame to file.
    
    Args:
        df: pandas DataFrame to save
        output_path: Path to save the file
        format: File format ('csv', 'excel', 'json')
    """
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'excel':
        df.to_excel(output_path, index=False)
    elif format == 'json':
        df.to_json(output_path, orient='records', indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Data saved to {output_path} in {format} format")