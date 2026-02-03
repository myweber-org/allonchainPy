import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
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

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by converting to appropriate types.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of column names to clean.
    
    Returns:
    pd.DataFrame: DataFrame with cleaned numeric columns.
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def validate_dataframe(df, required_columns):
    """
    Validate that DataFrame contains required columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    required_columns (list): List of required column names.
    
    Returns:
    bool: True if all required columns are present.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return False
    
    return True

def clean_data_pipeline(df, config):
    """
    Execute a complete data cleaning pipeline.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    config (dict): Configuration dictionary with cleaning options.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    if not validate_dataframe(df, config.get('required_columns', [])):
        raise ValueError("DataFrame missing required columns")
    
    if config.get('remove_duplicates', False):
        df = remove_duplicates(
            df, 
            subset=config.get('duplicate_subset'),
            keep=config.get('duplicate_keep', 'first')
        )
    
    if config.get('clean_numeric', False):
        df = clean_numeric_columns(
            df, 
            config.get('numeric_columns', [])
        )
    
    return df