import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    """
    if subset is None:
        subset = df.columns.tolist()
    return df.drop_duplicates(subset=subset, keep='first')

def convert_column_types(df, column_type_map):
    """
    Convert specified columns to given data types.
    """
    for column, dtype in column_type_map.items():
        if column in df.columns:
            df[column] = df[column].astype(dtype)
    return df

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy.
    """
    if columns is None:
        columns = df.columns[df.isnull().any()].tolist()
    
    for column in columns:
        if strategy == 'mean' and pd.api.types.is_numeric_dtype(df[column]):
            df[column].fillna(df[column].mean(), inplace=True)
        elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[column]):
            df[column].fillna(df[column].median(), inplace=True)
        elif strategy == 'mode':
            df[column].fillna(df[column].mode()[0], inplace=True)
        elif strategy == 'drop':
            df.dropna(subset=[column], inplace=True)
    
    return df

def normalize_column(df, column, method='minmax'):
    """
    Normalize specified column using given method.
    """
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val != min_val:
            df[column] = (df[column] - min_val) / (max_val - min_val)
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val != 0:
            df[column] = (df[column] - mean_val) / std_val
    
    return df

def clean_dataframe(df, operations):
    """
    Apply multiple cleaning operations to DataFrame.
    """
    for operation in operations:
        if operation['type'] == 'remove_duplicates':
            df = remove_duplicates(df, operation.get('subset'))
        elif operation['type'] == 'convert_types':
            df = convert_column_types(df, operation['type_map'])
        elif operation['type'] == 'handle_missing':
            df = handle_missing_values(
                df, 
                operation.get('strategy', 'mean'),
                operation.get('columns')
            )
        elif operation['type'] == 'normalize':
            df = normalize_column(
                df,
                operation['column'],
                operation.get('method', 'minmax')
            )
    
    return df
def remove_duplicates_preserve_order(seq):
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result