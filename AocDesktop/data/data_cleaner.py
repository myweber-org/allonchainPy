
import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for duplicates
        keep: determines which duplicates to mark ('first', 'last', False)
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame columns.
    
    Args:
        df: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'drop'
        columns: list of columns to process, None for all numeric columns
    
    Returns:
        DataFrame with missing values handled
    """
    df_copy = df.copy()
    
    if columns is None:
        columns = df_copy.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        return df_copy.dropna(subset=columns)
    
    for col in columns:
        if col in df_copy.columns:
            if strategy == 'mean':
                fill_value = df_copy[col].mean()
            elif strategy == 'median':
                fill_value = df_copy[col].median()
            elif strategy == 'mode':
                fill_value = df_copy[col].mode()[0] if not df_copy[col].mode().empty else 0
            else:
                fill_value = 0
            
            df_copy[col] = df_copy[col].fillna(fill_value)
    
    return df_copy

def normalize_column(df, column, method='minmax'):
    """
    Normalize values in a column.
    
    Args:
        df: pandas DataFrame
        column: column name to normalize
        method: 'minmax' or 'zscore'
    
    Returns:
        DataFrame with normalized column
    """
    df_copy = df.copy()
    
    if column not in df_copy.columns:
        return df_copy
    
    if method == 'minmax':
        col_min = df_copy[column].min()
        col_max = df_copy[column].max()
        if col_max != col_min:
            df_copy[column] = (df_copy[column] - col_min) / (col_max - col_min)
    
    elif method == 'zscore':
        col_mean = df_copy[column].mean()
        col_std = df_copy[column].std()
        if col_std != 0:
            df_copy[column] = (df_copy[column] - col_mean) / col_std
    
    return df_copy

def clean_dataframe(df, operations=None):
    """
    Apply multiple cleaning operations to DataFrame.
    
    Args:
        df: pandas DataFrame
        operations: list of tuples (operation, kwargs)
    
    Returns:
        Cleaned DataFrame
    """
    if operations is None:
        operations = [
            ('remove_duplicates', {}),
            ('handle_missing_values', {'strategy': 'mean'})
        ]
    
    result = df.copy()
    
    for operation, kwargs in operations:
        if operation == 'remove_duplicates':
            result = remove_duplicates(result, **kwargs)
        elif operation == 'handle_missing_values':
            result = handle_missing_values(result, **kwargs)
        elif operation == 'normalize_column' and 'column' in kwargs:
            result = normalize_column(result, **kwargs)
    
    return result

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10, 20, np.nan, 40, 50, 50],
        'score': [100, 200, 300, 400, 500, 500]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataframe(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)