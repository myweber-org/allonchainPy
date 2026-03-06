
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None, remove_duplicates=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.
    """
    df_clean = df.copy()
    
    if remove_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
        removed = initial_rows - len(df_clean)
        print(f"Removed {removed} duplicate rows.")
    
    if columns_to_clean is None:
        columns_to_clean = df_clean.select_dtypes(include=['object']).columns
    
    for col in columns_to_clean:
        if col in df_clean.columns and df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].apply(lambda x: normalize_string(x) if pd.notnull(x) else x)
            print(f"Normalized column: {col}")
    
    return df_clean

def normalize_string(s):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping special characters (except basic punctuation).
    """
    if not isinstance(s, str):
        return s
    
    s = s.lower().strip()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[^\w\s.,!?-]', '', s)
    
    return s

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    Returns a Series with boolean values indicating valid emails.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame.")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return df[email_column].apply(lambda x: bool(re.match(email_pattern, str(x))) if pd.notnull(x) else False)
def remove_duplicates(data_list):
    """
    Remove duplicate entries from a list while preserving order.
    Returns a new list with unique elements.
    """
    seen = set()
    unique_list = []
    for item in data_list:
        if item not in seen:
            seen.add(item)
            unique_list.append(item)
    return unique_list

def clean_numeric_data(values, default=0):
    """
    Clean numeric data by converting strings to floats.
    Non-numeric values are replaced with the default value.
    """
    cleaned = []
    for val in values:
        try:
            cleaned.append(float(val))
        except (ValueError, TypeError):
            cleaned.append(default)
    return cleaned

def filter_by_threshold(data, threshold, keep_above=True):
    """
    Filter data based on a threshold value.
    If keep_above is True, keep values above threshold.
    If keep_above is False, keep values below or equal to threshold.
    """
    if keep_above:
        return [x for x in data if x > threshold]
    else:
        return [x for x in data if x <= threshold]