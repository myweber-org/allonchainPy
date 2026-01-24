
import pandas as pd

def clean_dataframe(df, text_columns=None):
    """
    Clean a pandas DataFrame by removing rows with null values
    and standardizing text columns to lowercase.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    text_columns (list): List of column names to standardize as text.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Remove rows with any null values
    cleaned_df = cleaned_df.dropna()
    
    # Standardize text columns to lowercase
    if text_columns:
        for col in text_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].astype(str).str.lower().str.strip()
    
    return cleaned_df

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    subset (list): Columns to consider for identifying duplicates.
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed.
    """
    return df.drop_duplicates(subset=subset, keep='first')