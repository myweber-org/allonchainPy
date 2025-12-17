import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    columns_to_check (list): List of column names to check for duplicates
    drop_duplicates (bool): Whether to drop duplicate rows
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if drop_duplicates:
        if columns_to_check:
            df_clean = df_clean.drop_duplicates(subset=columns_to_check)
        else:
            df_clean = df_clean.drop_duplicates()
    
    if fill_missing == 'drop':
        df_clean = df_clean.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    elif fill_missing == 'mode':
        for col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else None)
    
    return df_clean

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

def remove_outliers_iqr(df, columns, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to check for outliers
    multiplier (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    for col in columns:
        if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean

def standardize_columns(df, column_mapping):
    """
    Standardize column names according to a mapping dictionary.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column_mapping (dict): Dictionary mapping old column names to new ones
    
    Returns:
    pd.DataFrame: DataFrame with standardized column names
    """
    return df.rename(columns=column_mapping)

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10.5, 20.3, np.nan, 15.7, 1000.0, 10.5],
        'category': ['A', 'B', 'A', 'C', 'A', 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned_df = clean_dataset(df, columns_to_check=['id'], fill_missing='mean')
    print(cleaned_df)
    
    is_valid, message = validate_dataframe(cleaned_df, required_columns=['id', 'value'])
    print(f"\nValidation: {message}")
import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    subset (list, optional): Columns to consider for duplicates
    keep (str): Which duplicates to keep - 'first', 'last', or False
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def handle_missing_values(df, strategy='drop', fill_value=None):
    """
    Handle missing values in DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): 'drop' to remove rows, 'fill' to fill values
    fill_value: Value to fill missing entries with
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    if df.empty:
        return df
    
    if strategy == 'drop':
        cleaned_df = df.dropna()
        removed_count = len(df) - len(cleaned_df)
        print(f"Removed {removed_count} rows with missing values")
    elif strategy == 'fill':
        if fill_value is None:
            fill_value = df.mean(numeric_only=True)
        cleaned_df = df.fillna(fill_value)
        print(f"Filled missing values with {fill_value}")
    else:
        raise ValueError("Strategy must be 'drop' or 'fill'")
    
    return cleaned_df

def validate_data_types(df, expected_types):
    """
    Validate column data types.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    expected_types (dict): Dictionary mapping columns to expected dtypes
    
    Returns:
    bool: True if all types match, False otherwise
    """
    mismatches = []
    
    for column, expected_type in expected_types.items():
        if column in df.columns:
            actual_type = df[column].dtype
            if actual_type != expected_type:
                mismatches.append((column, expected_type, actual_type))
    
    if mismatches:
        print("Data type mismatches found:")
        for column, expected, actual in mismatches:
            print(f"  {column}: expected {expected}, got {actual}")
        return False
    
    print("All data types are valid")
    return True

def clean_dataframe(df, cleaning_steps):
    """
    Apply multiple cleaning steps to DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    cleaning_steps (list): List of cleaning functions and their arguments
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for step in cleaning_steps:
        func = step['function']
        args = step.get('args', [])
        kwargs = step.get('kwargs', {})
        
        cleaned_df = func(cleaned_df, *args, **kwargs)
    
    return cleaned_df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 1, 4, 5, 3],
        'name': ['Alice', 'Bob', 'Charlie', 'Alice', 'David', 'Eve', 'Charlie'],
        'age': [25, 30, 35, 25, 28, np.nan, 35],
        'score': [85.5, 92.0, 78.5, 85.5, 88.0, 91.5, 78.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Define cleaning steps
    steps = [
        {
            'function': remove_duplicates,
            'kwargs': {'subset': ['id', 'name'], 'keep': 'first'}
        },
        {
            'function': handle_missing_values,
            'kwargs': {'strategy': 'fill', 'fill_value': 0}
        }
    ]
    
    # Apply cleaning
    cleaned_df = clean_dataframe(df, steps)
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)