
import re

def clean_string(text):
    """
    Cleans a string by:
    1. Removing leading and trailing whitespace.
    2. Converting multiple spaces/newlines/tabs to a single space.
    3. Converting the string to lowercase.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")

    # Strip leading/trailing whitespace
    text = text.strip()

    # Replace any sequence of whitespace characters with a single space
    text = re.sub(r'\s+', ' ', text)

    # Convert to lowercase
    text = text.lower()

    return text

def normalize_list(string_list):
    """
    Applies clean_string to each element in a list.
    Returns a new list with cleaned strings.
    """
    if not isinstance(string_list, list):
        raise TypeError("Input must be a list")

    return [clean_string(item) for item in string_list]import pandas as pd

def clean_dataset(df, drop_na=True, column_case='lower'):
    """
    Clean a pandas DataFrame by handling missing values and standardizing column names.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_na (bool): If True, drop rows with any null values. If False, fill with column mean.
    column_case (str): Desired case for column names ('lower', 'upper', or 'title')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if drop_na:
        cleaned_df = cleaned_df.dropna()
    else:
        for col in cleaned_df.select_dtypes(include=['number']).columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
    
    # Standardize column names
    if column_case == 'lower':
        cleaned_df.columns = cleaned_df.columns.str.lower()
    elif column_case == 'upper':
        cleaned_df.columns = cleaned_df.columns.str.upper()
    elif column_case == 'title':
        cleaned_df.columns = cleaned_df.columns.str.title()
    
    # Remove leading/trailing whitespace from string columns
    for col in cleaned_df.select_dtypes(include=['object']).columns:
        cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate dataset structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of column names that must be present
    
    Returns:
    dict: Dictionary with validation results
    """
    validation_results = {
        'row_count': len(df),
        'column_count': len(df.columns),
        'null_count': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_columns
        validation_results['all_required_present'] = len(missing_columns) == 0
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'Name': ['Alice', 'Bob', None, 'David'],
        'Age': [25, 30, None, 35],
        'Score': [85.5, 92.0, 78.5, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame (drop NA):")
    cleaned = clean_dataset(df, drop_na=True, column_case='lower')
    print(cleaned)
    print("\nValidation Results:")
    print(validate_dataset(cleaned, required_columns=['name', 'age']))