import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): If True, remove duplicate rows.
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            elif fill_missing == 'median':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            mode_val = cleaned_df[col].mode()
            if not mode_val.empty:
                cleaned_df[col] = cleaned_df[col].fillna(mode_val.iloc[0])
    
    return cleaned_df
import pandas as pd
import re

def clean_string_column(series):
    """Standardize string column: lowercase, strip, remove extra spaces."""
    if series.dtype == 'object':
        series = series.astype(str)
        series = series.str.lower()
        series = series.str.strip()
        series = series.apply(lambda x: re.sub(r'\s+', ' ', x))
    return series

def remove_duplicates(df, subset=None, keep='first'):
    """Remove duplicate rows from DataFrame."""
    return df.drop_duplicates(subset=subset, keep=keep)

def clean_dataframe(df, string_columns=None):
    """Apply cleaning functions to DataFrame."""
    df_clean = df.copy()
    
    if string_columns:
        for col in string_columns:
            if col in df_clean.columns:
                df_clean[col] = clean_string_column(df_clean[col])
    else:
        for col in df_clean.select_dtypes(include=['object']).columns:
            df_clean[col] = clean_string_column(df_clean[col])
    
    df_clean = remove_duplicates(df_clean)
    return df_clean

def validate_email(email):
    """Basic email validation using regex."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email)))