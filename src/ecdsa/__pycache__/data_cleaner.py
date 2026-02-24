
import re
import pandas as pd
from typing import Union, List, Optional

def remove_duplicates(data: Union[List, pd.Series, pd.DataFrame]) -> Union[List, pd.Series, pd.DataFrame]:
    """
    Remove duplicate entries from a list, Series, or DataFrame.
    """
    if isinstance(data, list):
        return list(dict.fromkeys(data))
    elif isinstance(data, pd.Series):
        return data.drop_duplicates()
    elif isinstance(data, pd.DataFrame):
        return data.drop_duplicates()
    else:
        raise TypeError("Input must be a list, pandas Series, or pandas DataFrame")

def validate_email(email: str) -> bool:
    """
    Validate an email address format using regex.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def normalize_text(text: str, lower_case: bool = True, remove_punctuation: bool = False) -> str:
    """
    Normalize text by converting to lowercase and optionally removing punctuation.
    """
    if lower_case:
        text = text.lower()
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def fill_missing_values(df: pd.DataFrame, column: str, fill_value: Optional[Union[str, int, float]] = None) -> pd.DataFrame:
    """
    Fill missing values in a DataFrame column with a specified value or the column mean.
    """
    df_copy = df.copy()
    if fill_value is not None:
        df_copy[column] = df_copy[column].fillna(fill_value)
    else:
        if pd.api.types.is_numeric_dtype(df_copy[column]):
            df_copy[column] = df_copy[column].fillna(df_copy[column].mean())
        else:
            df_copy[column] = df_copy[column].fillna('')
    return df_copy

def convert_date_format(date_str: str, input_format: str = '%Y-%m-%d', output_format: str = '%d/%m/%Y') -> str:
    """
    Convert a date string from one format to another.
    """
    try:
        date_obj = pd.to_datetime(date_str, format=input_format)
        return date_obj.strftime(output_format)
    except (ValueError, TypeError):
        return date_str

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names by removing whitespace, converting to lowercase, and replacing spaces with underscores.
    """
    df_copy = df.copy()
    df_copy.columns = [col.strip().lower().replace(' ', '_') for col in df_copy.columns]
    return df_copyimport pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str): Strategy to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            elif fill_missing == 'median':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            mode_val = cleaned_df[col].mode()
            if not mode_val.empty:
                cleaned_df[col] = cleaned_df[col].fillna(mode_val[0])
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for required columns and data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    return True, "Dataset is valid"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 3, None],
        'B': [4, None, 6, 7, 8],
        'C': ['x', 'y', 'x', 'z', 'y']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_dataset(cleaned, required_columns=['A', 'B'])
    print(f"\nValidation: {message}")