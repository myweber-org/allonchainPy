import pandas as pd

def clean_dataframe(df, drop_na=True, rename_columns=True):
    """
    Clean a pandas DataFrame by removing null values and standardizing column names.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_na (bool): If True, drop rows with any null values.
        rename_columns (bool): If True, rename columns to lowercase with underscores.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_na:
        cleaned_df = cleaned_df.dropna()
    
    if rename_columns:
        cleaned_df.columns = (
            cleaned_df.columns
            .str.lower()
            .str.replace(r'[^a-z0-9]+', '_', regex=True)
            .str.strip('_')
        )
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and required columns.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input must be a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"