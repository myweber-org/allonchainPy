
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
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    cleaned_df = df.copy()
    for col in columns:
        if col in cleaned_df.columns:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    return cleaned_df

def normalize_data(df, columns, method='minmax'):
    normalized_df = df.copy()
    for col in columns:
        if col in normalized_df.columns:
            if method == 'minmax':
                min_val = normalized_df[col].min()
                max_val = normalized_df[col].max()
                if max_val != min_val:
                    normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
                else:
                    normalized_df[col] = 0
            elif method == 'zscore':
                mean_val = normalized_df[col].mean()
                std_val = normalized_df[col].std()
                if std_val != 0:
                    normalized_df[col] = (normalized_df[col] - mean_val) / std_val
                else:
                    normalized_df[col] = 0
    return normalized_df

def clean_dataset(df, numeric_columns):
    df_cleaned = df.dropna(subset=numeric_columns)
    df_no_outliers = remove_outliers_iqr(df_cleaned, numeric_columns)
    df_normalized = normalize_data(df_no_outliers, numeric_columns, method='minmax')
    return df_normalized

if __name__ == "__main__":
    sample_data = {
        'feature_a': [1, 2, 3, 4, 100, 6, 7, 8, 9, 10],
        'feature_b': [10, 20, 30, 40, 50, 60, 70, 80, 90, 1000],
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    numeric_cols = ['feature_a', 'feature_b']
    
    cleaned_df = clean_dataset(df, numeric_cols)
    print("Original dataset shape:", df.shape)
    print("Cleaned dataset shape:", cleaned_df.shape)
    print("\nCleaned dataset:")
    print(cleaned_df)