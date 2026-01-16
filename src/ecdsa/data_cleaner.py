import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=True, fill_value=0):
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

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    return True, "DataFrame is valid"

def process_data_file(file_path, output_path=None):
    """
    Process a data file by cleaning and validating it.
    
    Parameters:
    file_path (str): Path to input data file.
    output_path (str): Path to save cleaned data.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read file: {e}")
    
    is_valid, message = validate_dataframe(df)
    if not is_valid:
        raise ValueError(f"Data validation failed: {message}")
    
    cleaned_df = clean_dataframe(df)
    
    if output_path:
        cleaned_df.to_csv(output_path, index=False)
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, 2, None, 5],
        'B': [None, 2, 2, 4, 5],
        'C': [1, 2, 3, 4, 5]
    })
    
    print("Original DataFrame:")
    print(sample_data)
    
    cleaned = clean_dataframe(sample_data)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned, ['A', 'B', 'C'])
    print(f"\nValidation: {message}")