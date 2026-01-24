import pandas as pd

def clean_dataset(df, missing_strategy='drop', duplicate_action='drop_first'):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    missing_strategy (str): Strategy for missing values - 'drop', 'fill_mean', 'fill_median', 'fill_mode'
    duplicate_action (str): Action for duplicates - 'drop_first', 'drop_last', 'keep_none'
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif missing_strategy == 'fill_mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
    elif missing_strategy == 'fill_median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
    elif missing_strategy == 'fill_mode':
        cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
    
    # Handle duplicates
    if duplicate_action == 'drop_first':
        cleaned_df = cleaned_df.drop_duplicates(keep='first')
    elif duplicate_action == 'drop_last':
        cleaned_df = cleaned_df.drop_duplicates(keep='last')
    elif duplicate_action == 'keep_none':
        cleaned_df = cleaned_df.drop_duplicates(keep=False)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

def get_data_summary(df):
    """
    Generate summary statistics for a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_stats': df.describe().to_dict() if df.select_dtypes(include='number').shape[1] > 0 else {},
        'categorical_stats': df.select_dtypes(include='object').describe().to_dict() if df.select_dtypes(include='object').shape[1] > 0 else {}
    }
    return summary

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = pd.DataFrame({
#         'A': [1, 2, None, 4, 5],
#         'B': [5, 6, 7, 7, 8],
#         'C': ['x', 'y', 'z', 'x', 'y']
#     })
#     
#     cleaned = clean_dataset(sample_data, missing_strategy='fill_mean', duplicate_action='drop_first')
#     print("Cleaned DataFrame:")
#     print(cleaned)
#     
#     is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
#     print(f"Validation: {is_valid} - {message}")
#     
#     summary = get_data_summary(cleaned)
#     print("Data Summary:")
#     print(summary)