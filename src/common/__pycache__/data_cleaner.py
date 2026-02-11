import pandas as pd
import re

def clean_text_column(df, column_name):
    """
    Standardize text by converting to lowercase, removing extra spaces,
    and eliminating special characters.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = df[column_name].astype(str)
    df[column_name] = df[column_name].str.lower()
    df[column_name] = df[column_name].str.strip()
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    return df

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def standardize_data(df, text_columns=None):
    """
    Apply cleaning operations to multiple text columns.
    """
    df_clean = df.copy()
    
    if text_columns:
        for col in text_columns:
            df_clean = clean_text_column(df_clean, col)
    
    df_clean = remove_duplicates(df_clean)
    return df_clean

if __name__ == "__main__":
    sample_data = {
        'name': ['John Doe', 'Jane Smith', 'JOHN DOE', 'Alice Johnson  ', 'bob@example'],
        'email': ['john@test.com', 'jane@test.com', 'JOHN@TEST.COM', 'alice@test.com', 'bob@test.com'],
        'age': [25, 30, 25, 28, 35]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = standardize_data(df, text_columns=['name', 'email'])
    print("\nCleaned DataFrame:")
    print(cleaned_df)