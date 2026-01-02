
import pandas as pd

def clean_dataset(df, text_columns=None, drop_na=True):
    """
    Clean a pandas DataFrame by handling missing values and standardizing text.
    
    Args:
        df: pandas DataFrame to clean
        text_columns: list of column names containing text data
        drop_na: if True, drop rows with any null values
    
    Returns:
        Cleaned pandas DataFrame
    """
    df_clean = df.copy()
    
    if drop_na:
        df_clean = df_clean.dropna()
    
    if text_columns:
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of required column names
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    sample_data = {
        'name': ['Alice', 'Bob', None, 'Charlie'],
        'age': [25, 30, 35, None],
        'city': ['New York', 'los angeles', 'Chicago', 'BOSTON']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned_df = clean_dataset(df, text_columns=['city'], drop_na=True)
    print(cleaned_df)
    
    is_valid, message = validate_dataframe(cleaned_df, required_columns=['name', 'age'])
    print(f"\nValidation: {is_valid} - {message}")