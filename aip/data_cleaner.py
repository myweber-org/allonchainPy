
import pandas as pd
import re

def clean_text_column(df, column_name):
    """
    Standardize text by converting to lowercase, removing extra spaces,
    and stripping special characters except alphanumeric and basic punctuation.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = df[column_name].astype(str).str.lower()
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'\s+', ' ', x.strip()))
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'[^\w\s.,!?-]', '', x))
    return df

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def standardize_dates(df, column_name, date_format='%Y-%m-%d'):
    """
    Attempt to parse and standardize date column to specified format.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = pd.to_datetime(df[column_name], errors='coerce').dt.strftime(date_format)
    return df

def clean_dataset(df, text_columns=None, date_columns=None, deduplicate=True):
    """
    Main function to clean dataset with multiple operations.
    """
    df_clean = df.copy()
    
    if deduplicate:
        df_clean = remove_duplicates(df_clean)
    
    if text_columns:
        for col in text_columns:
            if col in df_clean.columns:
                df_clean = clean_text_column(df_clean, col)
    
    if date_columns:
        for col in date_columns:
            if col in df_clean.columns:
                df_clean = standardize_dates(df_clean, col)
    
    return df_clean