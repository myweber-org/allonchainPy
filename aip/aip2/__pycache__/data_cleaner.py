
import pandas as pd
import numpy as np
from typing import List, Optional

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names by converting to lowercase and replacing spaces with underscores.
    """
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df

def remove_duplicate_rows(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from the DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df: pd.DataFrame, strategy: str = 'mean', columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Fill missing values in specified columns using the given strategy.
    """
    df_filled = df.copy()
    
    if columns is None:
        columns = df_filled.columns
    
    for col in columns:
        if df_filled[col].dtype in [np.float64, np.int64]:
            if strategy == 'mean':
                df_filled[col].fillna(df_filled[col].mean(), inplace=True)
            elif strategy == 'median':
                df_filled[col].fillna(df_filled[col].median(), inplace=True)
            elif strategy == 'mode':
                df_filled[col].fillna(df_filled[col].mode()[0], inplace=True)
            elif strategy == 'zero':
                df_filled[col].fillna(0, inplace=True)
    
    return df_filled

def standardize_text_columns(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Standardize text columns by stripping whitespace and converting to lowercase.
    """
    df_standardized = df.copy()
    
    if columns is None:
        columns = [col for col in df_standardized.columns if df_standardized[col].dtype == 'object']
    
    for col in columns:
        df_standardized[col] = df_standardized[col].astype(str).str.strip().str.lower()
    
    return df_standardized

def clean_dataframe(df: pd.DataFrame, 
                    clean_names: bool = True,
                    remove_duplicates: bool = True,
                    fill_missing: bool = True,
                    standardize_text: bool = True) -> pd.DataFrame:
    """
    Main function to clean a DataFrame by applying multiple cleaning steps.
    """
    cleaned_df = df.copy()
    
    if clean_names:
        cleaned_df = clean_column_names(cleaned_df)
    
    if remove_duplicates:
        cleaned_df = remove_duplicate_rows(cleaned_df)
    
    if fill_missing:
        cleaned_df = fill_missing_values(cleaned_df)
    
    if standardize_text:
        cleaned_df = standardize_text_columns(cleaned_df)
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'Name': ['Alice', 'Bob', 'Alice', 'Charlie', None],
        'Age': [25, 30, 25, 35, 40],
        'Score': [85.5, 90.0, 85.5, None, 95.0],
        'City': ['New York', 'Los Angeles', 'new york', 'Chicago', 'Chicago ']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned = clean_dataframe(df)
    print(cleaned)