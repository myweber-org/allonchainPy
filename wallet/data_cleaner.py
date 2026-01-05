import pandas as pd
import numpy as np

def remove_duplicates(df):
    """Remove duplicate rows from DataFrame."""
    return df.drop_duplicates()

def fill_missing_values(df, strategy='mean'):
    """Fill missing values using specified strategy."""
    if strategy == 'mean':
        return df.fillna(df.mean(numeric_only=True))
    elif strategy == 'median':
        return df.fillna(df.median(numeric_only=True))
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    else:
        return df.fillna(0)

def normalize_column(df, column_name):
    """Normalize specified column to range [0,1]."""
    if column_name in df.columns:
        col_min = df[column_name].min()
        col_max = df[column_name].max()
        if col_max != col_min:
            df[column_name] = (df[column_name] - col_min) / (col_max - col_min)
    return df

def clean_dataframe(df, remove_dups=True, fill_strategy='mean', normalize_cols=None):
    """Apply multiple cleaning operations to DataFrame."""
    if remove_dups:
        df = remove_duplicates(df)
    
    df = fill_missing_values(df, strategy=fill_strategy)
    
    if normalize_cols:
        for col in normalize_cols:
            if col in df.columns:
                df = normalize_column(df, col)
    
    return df

def load_and_clean_csv(filepath, **kwargs):
    """Load CSV file and apply cleaning operations."""
    try:
        df = pd.read_csv(filepath)
        return clean_dataframe(df, **kwargs)
    except Exception as e:
        print(f"Error loading or cleaning file: {e}")
        return None