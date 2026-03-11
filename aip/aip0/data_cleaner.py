
import pandas as pd
import numpy as np

def clean_csv_data(file_path, missing_strategy='mean', columns_to_drop=None):
    """
    Load and clean CSV data by handling missing values and optionally dropping columns.
    
    Parameters:
    file_path (str): Path to the CSV file.
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop').
    columns_to_drop (list): List of column names to drop.
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading file: {e}")
    
    original_shape = df.shape
    
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop, errors='ignore')
    
    if missing_strategy == 'drop':
        df = df.dropna()
    elif missing_strategy in ['mean', 'median']:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if missing_strategy == 'mean':
                fill_value = df[col].mean()
            else:
                fill_value = df[col].median()
            df[col] = df[col].fillna(fill_value)
    elif missing_strategy == 'mode':
        for col in df.columns:
            mode_value = df[col].mode()
            if not mode_value.empty:
                df[col] = df[col].fillna(mode_value[0])
    
    cleaned_shape = df.shape
    print(f"Original data shape: {original_shape}")
    print(f"Cleaned data shape: {cleaned_shape}")
    print(f"Rows removed: {original_shape[0] - cleaned_shape[0]}")
    print(f"Columns removed: {original_shape[1] - cleaned_shape[1]}")
    
    return df

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to a CSV file.
    
    Parameters:
    df (pandas.DataFrame): DataFrame to save.
    output_path (str): Path for the output CSV file.
    """
    try:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    except Exception as e:
        raise Exception(f"Error saving file: {e}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    try:
        cleaned_df = clean_csv_data(
            file_path=input_file,
            missing_strategy='mean',
            columns_to_drop=['unnecessary_column']
        )
        save_cleaned_data(cleaned_df, output_file)
    except Exception as e:
        print(f"Error during data cleaning: {e}")