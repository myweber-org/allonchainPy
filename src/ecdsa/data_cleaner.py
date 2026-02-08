
import pandas as pd
import numpy as np

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, fill_missing=True):
    """
    Clean a pandas DataFrame by standardizing columns, removing duplicates,
    and handling missing values.
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        for col in cleaned_df.select_dtypes(include=[np.number]).columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[col] = cleaned_df[col].fillna('Unknown')
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

def validate_dataframe(df, required_columns=None, unique_columns=None):
    """
    Validate DataFrame structure and data integrity.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if unique_columns:
        for col in unique_columns:
            if col in df.columns and df[col].duplicated().any():
                raise ValueError(f"Column '{col}' contains duplicate values")
    
    return True

def process_csv_file(input_path, output_path, **kwargs):
    """
    Read CSV file, clean data, and save to output path.
    """
    df = pd.read_csv(input_path)
    cleaned_df = clean_dataframe(df, **kwargs)
    cleaned_df.to_csv(output_path, index=False)
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Alice', None, 'Charlie'],
        'Age': [25, 30, 25, 40, None],
        'City': ['NYC', 'LA', 'NYC', 'Chicago', 'Boston']
    })
    
    cleaned = clean_dataframe(sample_data)
    print("Original DataFrame:")
    print(sample_data)
    print("\nCleaned DataFrame:")
    print(cleaned)