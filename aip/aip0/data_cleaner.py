import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (bool): Whether to fill missing values.
    fill_value: Value to use for filling missing data.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"

def normalize_column(df, column_name):
    """
    Normalize a column to have values between 0 and 1.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column_name (str): Name of column to normalize.
    
    Returns:
    pd.DataFrame: DataFrame with normalized column.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    normalized_df = df.copy()
    col_min = normalized_df[column_name].min()
    col_max = normalized_df[column_name].max()
    
    if col_max == col_min:
        normalized_df[column_name] = 0.5
    else:
        normalized_df[column_name] = (normalized_df[column_name] - col_min) / (col_max - col_min)
    
    return normalized_df