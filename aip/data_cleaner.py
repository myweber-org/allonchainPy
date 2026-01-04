import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (list, optional): Column names to consider for duplicates
        keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    if subset is not None:
        if not all(col in df.columns for col in subset):
            raise ValueError("All subset columns must exist in DataFrame")
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate row(s)")
    
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(df):
    """
    Basic validation of DataFrame structure.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
    
    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        print("Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("DataFrame is empty")
        return False
    
    return True

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by converting to appropriate dtype.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to clean
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns
    """
    if not validate_dataframe(df):
        return df
    
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', 'David', 'Eve'],
        'score': ['85', '90', '90', '78', '92', '92', '88']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nAfter removing duplicates:")
    cleaned_df = remove_duplicates(df, subset=['id', 'name'])
    print(cleaned_df)
    
    print("\nAfter cleaning numeric column:")
    cleaned_df = clean_numeric_columns(cleaned_df, ['score'])
    print(cleaned_df.dtypes)
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, columns=None):
    """
    Remove outliers from DataFrame using Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to process. If None, process all numeric columns.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def standardize_columns(df, columns=None):
    """
    Standardize specified columns to have zero mean and unit variance.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to standardize
    
    Returns:
    pd.DataFrame: DataFrame with standardized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_standardized = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        mean_val = df[col].mean()
        std_val = df[col].std()
        
        if std_val > 0:
            df_standardized[col] = (df[col] - mean_val) / std_val
    
    return df_standardized

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame using specified strategy.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Strategy to handle missing values ('mean', 'median', 'mode', 'drop')
    columns (list): List of column names to process
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    if columns is None:
        columns = df.columns.tolist()
    
    df_processed = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if strategy == 'drop':
            df_processed = df_processed.dropna(subset=[col])
        elif strategy == 'mean':
            df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
        elif strategy == 'median':
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        elif strategy == 'mode':
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
    
    return df_processed.reset_index(drop=True)import re

def clean_string(text):
    """
    Cleans and normalizes a string by:
    - Converting to lowercase.
    - Removing leading/trailing whitespace.
    - Replacing multiple spaces with a single space.
    - Removing non-alphanumeric characters (except basic punctuation).
    """
    if not isinstance(text, str):
        return text

    # Convert to lowercase and strip whitespace
    text = text.lower().strip()

    # Replace multiple spaces/newlines/tabs with a single space
    text = re.sub(r'\s+', ' ', text)

    # Remove non-alphanumeric characters, but keep basic punctuation
    # This keeps letters, numbers, spaces, and .,!?;:-
    text = re.sub(r'[^a-z0-9\s.,!?;:-]', '', text)

    return text

def normalize_whitespace(text):
    """
    Normalizes whitespace in a string by ensuring single spaces between words.
    """
    if not isinstance(text, str):
        return text
    return ' '.join(text.split())