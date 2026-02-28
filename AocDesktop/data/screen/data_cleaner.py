
import pandas as pd

def clean_dataset(df, id_column='id'):
    """
    Remove duplicate rows based on an ID column and standardize column names.
    """
    if df.empty:
        return df

    df_clean = df.copy()

    if id_column in df_clean.columns:
        df_clean = df_clean.drop_duplicates(subset=[id_column], keep='first')
    else:
        df_clean = df_clean.drop_duplicates()

    df_clean.columns = df_clean.columns.str.strip().str.lower().str.replace(' ', '_')
    df_clean = df_clean.reset_index(drop=True)

    return df_clean

def validate_numeric_columns(df, numeric_columns):
    """
    Ensure specified columns contain only numeric data, coerce errors to NaN.
    """
    df_valid = df.copy()
    for col in numeric_columns:
        if col in df_valid.columns:
            df_valid[col] = pd.to_numeric(df_valid[col], errors='coerce')
    return df_valid

if __name__ == "__main__":
    sample_data = {
        'ID': [1, 2, 2, 3, 4],
        'Customer Name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David'],
        'Order Value': ['100', '150', '150', 'two hundred', '300']
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)

    cleaned_df = clean_dataset(df, id_column='ID')
    cleaned_df = validate_numeric_columns(cleaned_df, ['Order Value'])
    print("\nCleaned DataFrame:")
    print(cleaned_df)