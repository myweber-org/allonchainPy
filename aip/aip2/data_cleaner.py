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
    Clean numeric columns by converting to appropriate dtype and handling errors.
    
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
    Execute a complete data cleaning pipeline based on configuration.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    config (dict): Configuration dictionary with cleaning options.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    if 'required_columns' in config:
        if not validate_dataframe(df, config['required_columns']):
            raise ValueError("DataFrame missing required columns")
    
    if 'remove_duplicates' in config and config['remove_duplicates']:
        subset = config.get('duplicate_subset', None)
        df = remove_duplicates(df, subset=subset)
    
    if 'clean_numeric' in config:
        df = clean_numeric_columns(df, config['clean_numeric'])
    
    return df