
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
    return df_copy