import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fillna_strategy='mean', columns_to_clean=None):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fillna_strategy (str): Strategy for filling NaN values ('mean', 'median', 'mode', or 'zero').
    columns_to_clean (list): Specific columns to apply cleaning. If None, all columns are cleaned.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    if columns_to_clean is None:
        columns_to_clean = df_clean.columns.tolist()
    
    if drop_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
        removed = initial_rows - len(df_clean)
        print(f"Removed {removed} duplicate rows.")
    
    for col in columns_to_clean:
        if col in df_clean.columns:
            if df_clean[col].dtype in ['int64', 'float64']:
                if fillna_strategy == 'mean':
                    fill_value = df_clean[col].mean()
                elif fillna_strategy == 'median':
                    fill_value = df_clean[col].median()
                elif fillna_strategy == 'mode':
                    fill_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 0
                elif fillna_strategy == 'zero':
                    fill_value = 0
                else:
                    raise ValueError("fillna_strategy must be 'mean', 'median', 'mode', or 'zero'")
                
                null_count = df_clean[col].isnull().sum()
                if null_count > 0:
                    df_clean[col] = df_clean[col].fillna(fill_value)
                    print(f"Filled {null_count} missing values in column '{col}' with {fillna_strategy} value {fill_value:.2f}")
            else:
                null_count = df_clean[col].isnull().sum()
                if null_count > 0:
                    df_clean[col] = df_clean[col].fillna('Unknown')
                    print(f"Filled {null_count} missing values in column '{col}' with 'Unknown'")
    
    return df_clean

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a specific column using the Interquartile Range (IQR) method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to process.
    multiplier (float): IQR multiplier for outlier detection.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    initial_len = len(df)
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].reset_index(drop=True)
    removed = initial_len - len(df_filtered)
    
    print(f"Removed {removed} outliers from column '{column}' using IQR method.")
    return df_filtered

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5, 6],
        'value': [10.5, 20.3, np.nan, 40.1, 50.0, 50.0, 1000.0],
        'category': ['A', 'B', None, 'A', 'B', 'B', 'C']
    }
    
    df_sample = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df_sample)
    print("\n" + "="*50 + "\n")
    
    df_cleaned = clean_dataset(df_sample, fillna_strategy='median')
    print("\nCleaned DataFrame:")
    print(df_cleaned)
    
    df_no_outliers = remove_outliers_iqr(df_cleaned, 'value')
    print("\nDataFrame after outlier removal:")
    print(df_no_outliers)import pandas as pd
import re

def clean_text_column(series):
    """
    Standardize text: lowercase, strip whitespace, remove extra spaces.
    """
    if series.dtype == 'object':
        series = series.astype(str)
        series = series.str.lower()
        series = series.str.strip()
        series = series.apply(lambda x: re.sub(r'\s+', ' ', x))
    return series

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def clean_dataframe(df, text_columns=None):
    """
    Apply cleaning functions to DataFrame.
    """
    df_clean = df.copy()
    
    if text_columns:
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = clean_text_column(df_clean[col])
    
    df_clean = remove_duplicates(df_clean)
    return df_clean

if __name__ == "__main__":
    sample_data = {
        'name': ['  John Doe  ', 'Jane SMITH', 'John Doe', 'Alice   Brown'],
        'email': ['JOHN@email.com', 'jane@email.com', 'john@email.com', 'alice@email.com'],
        'age': [25, 30, 25, 28]
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataframe(df, text_columns=['name', 'email'])
    print("\nCleaned DataFrame:")
    print(cleaned_df)