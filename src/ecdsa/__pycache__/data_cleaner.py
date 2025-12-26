import pandas as pd
import numpy as np

def clean_missing_data(df, strategy='mean', columns=None):
    """
    Clean missing data in a pandas DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    strategy (str): Strategy for handling missing values.
                    Options: 'mean', 'median', 'mode', 'drop', 'fill'.
    columns (list): List of columns to apply cleaning. If None, applies to all numeric columns.

    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    if df.empty:
        return df

    df_cleaned = df.copy()

    if columns is None:
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols

    for col in columns:
        if col not in df_cleaned.columns:
            continue

        if strategy == 'mean':
            fill_value = df_cleaned[col].mean()
        elif strategy == 'median':
            fill_value = df_cleaned[col].median()
        elif strategy == 'mode':
            fill_value = df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else np.nan
        elif strategy == 'drop':
            df_cleaned = df_cleaned.dropna(subset=[col])
            continue
        elif strategy == 'fill':
            fill_value = 0
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        df_cleaned[col] = df_cleaned[col].fillna(fill_value)

    return df_cleaned

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.

    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.

    Returns:
    bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        return False

    if df.empty:
        return False

    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False

    return True

def load_and_clean_csv(file_path, **kwargs):
    """
    Load CSV file and clean missing data.

    Parameters:
    file_path (str): Path to CSV file.
    **kwargs: Additional arguments passed to clean_missing_data.

    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        if validate_dataframe(df):
            return clean_missing_data(df, **kwargs)
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"Error loading file: {e}")
        return pd.DataFrame()import pandas as pd
import numpy as np

def clean_missing_data(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame using specified strategy.
    
    Args:
        df: pandas DataFrame containing data with potential missing values
        strategy: Method for handling missing values ('mean', 'median', 'mode', 'drop')
        columns: List of columns to apply cleaning to (None applies to all columns)
    
    Returns:
        Cleaned pandas DataFrame
    """
    if df.empty:
        return df
    
    if columns is None:
        columns = df.columns
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if df_clean[col].isnull().sum() == 0:
            continue
        
        if strategy == 'drop':
            df_clean = df_clean.dropna(subset=[col])
        elif strategy == 'mean':
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        elif strategy == 'median':
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        elif strategy == 'mode':
            if not df_clean[col].mode().empty:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of column names that must be present
    
    Returns:
        Tuple of (is_valid, error_message)
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

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: Columns to consider for identifying duplicates
        keep: Which duplicates to keep ('first', 'last', False)
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def normalize_numeric_columns(df, columns=None):
    """
    Normalize numeric columns to range [0, 1].
    
    Args:
        df: pandas DataFrame
        columns: List of columns to normalize (None for all numeric columns)
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    
    for col in columns:
        if col in df_normalized.columns and pd.api.types.is_numeric_dtype(df_normalized[col]):
            col_min = df_normalized[col].min()
            col_max = df_normalized[col].max()
            
            if col_max > col_min:
                df_normalized[col] = (df_normalized[col] - col_min) / (col_max - col_min)
    
    return df_normalized

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, 4, 5],
        'C': [1, 1, 2, 2, 3]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame (mean imputation):")
    cleaned_df = clean_missing_data(df, strategy='mean')
    print(cleaned_df)