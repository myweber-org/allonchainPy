
import pandas as pd

def clean_dataset(df, missing_strategy='drop', duplicate_strategy='drop_first'):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    missing_strategy (str): Strategy for handling missing values.
        'drop': Drop rows with any missing values.
        'fill_mean': Fill missing values with column mean (numeric only).
        'fill_median': Fill missing values with column median (numeric only).
        'fill_mode': Fill missing values with column mode.
    duplicate_strategy (str): Strategy for handling duplicate rows.
        'drop_first': Drop duplicates keeping first occurrence.
        'drop_last': Drop duplicates keeping last occurrence.
        'keep_none': Drop all duplicates.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    # Handle missing values
    if missing_strategy == 'drop':
        df_clean = df_clean.dropna()
    elif missing_strategy == 'fill_mean':
        numeric_cols = df_clean.select_dtypes(include=['number']).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    elif missing_strategy == 'fill_median':
        numeric_cols = df_clean.select_dtypes(include=['number']).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    elif missing_strategy == 'fill_mode':
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                mode_val = df_clean[col].mode()
                if not mode_val.empty:
                    df_clean[col] = df_clean[col].fillna(mode_val[0])
    
    # Handle duplicates
    if duplicate_strategy == 'drop_first':
        df_clean = df_clean.drop_duplicates(keep='first')
    elif duplicate_strategy == 'drop_last':
        df_clean = df_clean.drop_duplicates(keep='last')
    elif duplicate_strategy == 'keep_none':
        df_clean = df_clean.drop_duplicates(keep=False)
    
    return df_clean

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    
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

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, None, 4, 5, 5],
        'B': [10, 20, 30, None, 50, 50],
        'C': ['x', 'y', 'z', 'x', None, 'x']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame (drop missing, drop first duplicate):")
    cleaned_df = clean_dataset(df, missing_strategy='drop', duplicate_strategy='drop_first')
    print(cleaned_df)
    
    is_valid, message = validate_dataset(cleaned_df, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {message}")