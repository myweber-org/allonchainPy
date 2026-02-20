
import pandas as pd

def clean_dataset(df, missing_strategy='drop', duplicate_strategy='drop_first'):
    """
    Clean a pandas DataFrame by handling missing values and removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    missing_strategy (str): Strategy for handling missing values.
        'drop': Drop rows with any missing values.
        'fill_mean': Fill missing values with column mean (numeric only).
        'fill_median': Fill missing values with column median (numeric only).
    duplicate_strategy (str): Strategy for handling duplicate rows.
        'drop_first': Drop duplicates, keeping first occurrence.
        'drop_last': Drop duplicates, keeping last occurrence.
        'keep_none': Drop all duplicate rows.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif missing_strategy == 'fill_mean':
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
    elif missing_strategy == 'fill_median':
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
    
    # Handle duplicates
    if duplicate_strategy == 'drop_first':
        cleaned_df = cleaned_df.drop_duplicates(keep='first')
    elif duplicate_strategy == 'drop_last':
        cleaned_df = cleaned_df.drop_duplicates(keep='last')
    elif duplicate_strategy == 'keep_none':
        cleaned_df = cleaned_df.drop_duplicates(keep=False)
    
    return cleaned_df

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate a DataFrame for basic integrity checks.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (is_valid, message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Dataset is valid"