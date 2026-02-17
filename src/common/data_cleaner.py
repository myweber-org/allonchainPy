
import re
import pandas as pd
from typing import Optional, List, Dict, Any

def remove_special_characters(text: str, keep_chars: str = '') -> str:
    """
    Remove special characters from a string, optionally keeping specified characters.

    Args:
        text: The input string.
        keep_chars: A string of characters to keep (e.g., '.-_').

    Returns:
        The cleaned string.
    """
    if not isinstance(text, str):
        return text
    pattern = f'[^a-zA-Z0-9\\s{re.escape(keep_chars)}]'
    return re.sub(pattern, '', text)

def validate_email(email: str) -> bool:
    """
    Validate an email address format.

    Args:
        email: The email address string.

    Returns:
        True if the email format is valid, False otherwise.
    """
    if not isinstance(email, str):
        return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def normalize_phone_number(phone: str, country_code: str = '+1') -> Optional[str]:
    """
    Normalize a phone number to a standard format.

    Args:
        phone: The input phone number string.
        country_code: The country code to prepend.

    Returns:
        The normalized phone number or None if invalid.
    """
    if not isinstance(phone, str):
        return None
    digits = re.sub(r'\D', '', phone)
    if len(digits) == 10:
        return f'{country_code}{digits}'
    elif len(digits) == 11 and digits.startswith('1'):
        return f'+{digits}'
    else:
        return None

def clean_dataframe(df: pd.DataFrame, column_rules: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply cleaning rules to specific columns in a DataFrame.

    Args:
        df: The input pandas DataFrame.
        column_rules: A dictionary mapping column names to cleaning functions.

    Returns:
        A cleaned DataFrame.
    """
    df_clean = df.copy()
    for column, func in column_rules.items():
        if column in df_clean.columns:
            df_clean[column] = df_clean[column].apply(func)
    return df_clean

def find_missing_values(df: pd.DataFrame) -> Dict[str, int]:
    """
    Count missing values for each column in a DataFrame.

    Args:
        df: The input pandas DataFrame.

    Returns:
        A dictionary with column names as keys and missing value counts as values.
    """
    missing = df.isnull().sum()
    return missing[missing > 0].to_dict()