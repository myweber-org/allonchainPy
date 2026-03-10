
import pandas as pd
import re

def clean_dataframe(df, column_names):
    """
    Clean a pandas DataFrame by removing duplicate rows and normalizing
    specified string columns (strip whitespace, lowercase).
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    
    # Normalize specified string columns
    for col in column_names:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].astype(str).apply(
                lambda x: re.sub(r'\s+', ' ', x.strip().lower())
            )
    
    return df_cleaned

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column and add a validation flag.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['email_valid'] = df[email_column].str.match(email_pattern, na=False)
    
    return df

def save_cleaned_data(df, output_path, format='csv'):
    """
    Save cleaned DataFrame to a file in specified format.
    """
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'excel':
        df.to_excel(output_path, index=False)
    elif format == 'json':
        df.to_json(output_path, orient='records')
    else:
        raise ValueError("Unsupported format. Choose 'csv', 'excel', or 'json'.")
    
    print(f"Data successfully saved to {output_path}")import re
from typing import List, Optional

def remove_special_characters(text: str, keep_spaces: bool = True) -> str:
    """
    Remove all special characters from the input text.
    
    Args:
        text: Input string to clean.
        keep_spaces: If True, preserve spaces. If False, remove spaces as well.
    
    Returns:
        Cleaned string with only alphanumeric characters and optionally spaces.
    """
    if keep_spaces:
        pattern = r'[^A-Za-z0-9\s]'
    else:
        pattern = r'[^A-Za-z0-9]'
    
    return re.sub(pattern, '', text)

def normalize_whitespace(text: str) -> str:
    """
    Replace multiple whitespace characters with a single space.
    
    Args:
        text: Input string to normalize.
    
    Returns:
        String with normalized whitespace.
    """
    return re.sub(r'\s+', ' ', text).strip()

def clean_text_pipeline(text: str, 
                       remove_special: bool = True,
                       normalize_ws: bool = True,
                       to_lower: bool = False) -> str:
    """
    Apply a series of cleaning operations to text.
    
    Args:
        text: Input string to process.
        remove_special: Whether to remove special characters.
        normalize_ws: Whether to normalize whitespace.
        to_lower: Whether to convert text to lowercase.
    
    Returns:
        Processed text after applying specified operations.
    """
    processed = text
    
    if remove_special:
        processed = remove_special_characters(processed)
    
    if normalize_ws:
        processed = normalize_whitespace(processed)
    
    if to_lower:
        processed = processed.lower()
    
    return processed

def batch_clean_texts(texts: List[str], **kwargs) -> List[str]:
    """
    Apply cleaning pipeline to a list of texts.
    
    Args:
        texts: List of strings to clean.
        **kwargs: Arguments to pass to clean_text_pipeline.
    
    Returns:
        List of cleaned strings.
    """
    return [clean_text_pipeline(text, **kwargs) for text in texts]

def extract_hashtags(text: str) -> List[str]:
    """
    Extract hashtags from social media style text.
    
    Args:
        text: Input string containing hashtags.
    
    Returns:
        List of hashtags found in the text.
    """
    return re.findall(r'#(\w+)', text)

def validate_email(email: str) -> Optional[str]:
    """
    Validate and return normalized email address if valid.
    
    Args:
        email: Email string to validate.
    
    Returns:
        Normalized email if valid, None otherwise.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if re.match(pattern, email):
        return email.strip().lower()
    return None