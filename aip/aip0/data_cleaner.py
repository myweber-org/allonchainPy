import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (bool): Whether to fill missing values.
    fill_value: Value to use for filling missing data.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"

def normalize_column(df, column_name):
    """
    Normalize a column to have values between 0 and 1.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column_name (str): Name of column to normalize.
    
    Returns:
    pd.DataFrame: DataFrame with normalized column.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    normalized_df = df.copy()
    col_min = normalized_df[column_name].min()
    col_max = normalized_df[column_name].max()
    
    if col_max == col_min:
        normalized_df[column_name] = 0.5
    else:
        normalized_df[column_name] = (normalized_df[column_name] - col_min) / (col_max - col_min)
    
    return normalized_df
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    cleaned_df = df.copy()
    for col in columns:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    return cleaned_df

def normalize_minmax(df, columns):
    normalized_df = df.copy()
    for col in columns:
        min_val = normalized_df[col].min()
        max_val = normalized_df[col].max()
        if max_val != min_val:
            normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
    return normalized_df

def clean_dataset(input_path, output_path, numeric_columns):
    try:
        df = pd.read_csv(input_path)
        df_cleaned = remove_outliers_iqr(df, numeric_columns)
        df_normalized = normalize_minmax(df_cleaned, numeric_columns)
        df_normalized.to_csv(output_path, index=False)
        print(f"Data cleaned and saved to {output_path}")
        print(f"Original rows: {len(df)}, Cleaned rows: {len(df_normalized)}")
        return df_normalized
    except Exception as e:
        print(f"Error during cleaning: {e}")
        return None

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    numeric_cols = ["age", "income", "score"]
    clean_dataset(input_file, output_file, numeric_cols)import pandas as pd
import re

def clean_text_column(df, column_name):
    """Standardize text by lowercasing and removing extra whitespace."""
    if column_name in df.columns:
        df[column_name] = df[column_name].astype(str).str.lower()
        df[column_name] = df[column_name].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    return df

def remove_duplicates(df, subset=None):
    """Remove duplicate rows from the DataFrame."""
    return df.drop_duplicates(subset=subset, keep='first')

def main():
    # Example usage
    data = {
        'name': ['Alice', 'alice', 'Bob  ', 'bob', 'Charlie'],
        'email': ['alice@example.com', 'alice@example.com', 'bob@example.com', 'bob@example.com', 'charlie@example.com']
    }
    df = pd.DataFrame(data)
    
    print("Original DataFrame:")
    print(df)
    
    df = clean_text_column(df, 'name')
    df = remove_duplicates(df, subset=['email'])
    
    print("\nCleaned DataFrame:")
    print(df)

if __name__ == "__main__":
    main()