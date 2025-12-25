import pandas as pd
import numpy as np

def clean_csv_data(filepath, strategy='mean', fill_value=None):
    """
    Clean a CSV file by handling missing values.
    
    Args:
        filepath (str): Path to the CSV file.
        strategy (str): Strategy for handling missing values.
            Options: 'mean', 'median', 'mode', 'constant', 'drop'.
        fill_value: Value to use when strategy is 'constant'.
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if df.empty:
        return df
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        df_cleaned = df.dropna()
    elif strategy == 'mean':
        df_cleaned = df.copy()
        for col in numeric_cols:
            df_cleaned[col].fillna(df[col].mean(), inplace=True)
    elif strategy == 'median':
        df_cleaned = df.copy()
        for col in numeric_cols:
            df_cleaned[col].fillna(df[col].median(), inplace=True)
    elif strategy == 'mode':
        df_cleaned = df.copy()
        for col in df.columns:
            mode_val = df[col].mode()
            if not mode_val.empty:
                df_cleaned[col].fillna(mode_val[0], inplace=True)
    elif strategy == 'constant':
        if fill_value is None:
            raise ValueError("fill_value must be provided for constant strategy")
        df_cleaned = df.fillna(fill_value)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return df_cleaned

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to CSV.
    
    Args:
        df (pandas.DataFrame): DataFrame to save.
        output_path (str): Path for output CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    input_file = "sample_data.csv"
    output_file = "cleaned_data.csv"
    
    try:
        cleaned_df = clean_csv_data(input_file, strategy='median')
        save_cleaned_data(cleaned_df, output_file)
        print(f"Original shape: {pd.read_csv(input_file).shape}")
        print(f"Cleaned shape: {cleaned_df.shape}")
    except Exception as e:
        print(f"Error during data cleaning: {e}")