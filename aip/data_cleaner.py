import pandas as pd
import numpy as np

def clean_missing_data(file_path, strategy='mean', columns=None):
    """
    Load a CSV file and handle missing values using specified strategy.
    
    Args:
        file_path (str): Path to the CSV file
        strategy (str): Method for handling missing values ('mean', 'median', 'mode', 'drop')
        columns (list): Specific columns to clean, if None cleans all columns
    
    Returns:
        pandas.DataFrame: Cleaned dataframe
    """
    try:
        df = pd.read_csv(file_path)
        
        if columns is None:
            columns = df.columns
        
        for column in columns:
            if column in df.columns:
                if df[column].isnull().any():
                    if strategy == 'mean':
                        df[column].fillna(df[column].mean(), inplace=True)
                    elif strategy == 'median':
                        df[column].fillna(df[column].median(), inplace=True)
                    elif strategy == 'mode':
                        df[column].fillna(df[column].mode()[0], inplace=True)
                    elif strategy == 'drop':
                        df.dropna(subset=[column], inplace=True)
        
        return df
    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Args:
        df (pandas.DataFrame): Dataframe to validate
        required_columns (list): List of columns that must be present
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if df is None or df.empty:
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

def save_cleaned_data(df, output_path):
    """
    Save cleaned dataframe to CSV file.
    
    Args:
        df (pandas.DataFrame): Cleaned dataframe
        output_path (str): Path to save the cleaned data
    
    Returns:
        bool: True if save successful, False otherwise
    """
    try:
        df.to_csv(output_path, index=False)
        return True
    except Exception as e:
        print(f"Error saving file: {str(e)}")
        return False
def remove_duplicates(data_list):
    """
    Remove duplicate entries from a list while preserving order.
    
    Args:
        data_list: List of items that may contain duplicates.
    
    Returns:
        List with duplicates removed.
    """
    seen = set()
    result = []
    
    for item in data_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    
    return result

def clean_numeric_data(values):
    """
    Clean numeric data by converting strings to floats and removing None values.
    
    Args:
        values: List of numeric values as strings or numbers.
    
    Returns:
        List of cleaned numeric values.
    """
    cleaned = []
    
    for value in values:
        if value is None:
            continue
        
        try:
            if isinstance(value, str):
                cleaned.append(float(value))
            else:
                cleaned.append(float(value))
        except (ValueError, TypeError):
            continue
    
    return cleaned

def validate_email_format(email):
    """
    Basic email format validation.
    
    Args:
        email: String to validate as email.
    
    Returns:
        Boolean indicating if email format is valid.
    """
    if not email or not isinstance(email, str):
        return False
    
    parts = email.split('@')
    
    if len(parts) != 2:
        return False
    
    if not parts[0] or not parts[1]:
        return False
    
    if '.' not in parts[1]:
        return False
    
    return True