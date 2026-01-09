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

def clean_dataset(file_path, output_path=None):
    """Main function to clean dataset with default operations."""
    try:
        df = pd.read_csv(file_path)
        df = remove_duplicates(df)
        df = fill_missing_values(df, strategy='mean')
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df = normalize_column(df, col)
        
        if output_path:
            df.to_csv(output_path, index=False)
            return f"Cleaned data saved to {output_path}"
        return df
    except Exception as e:
        return f"Error cleaning dataset: {str(e)}"