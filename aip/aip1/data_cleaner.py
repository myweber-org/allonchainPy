
import json

def clean_data(input_file, output_file, key='valid'):
    """
    Load JSON data from input_file, filter entries where the specified key
    is True, and save the cleaned data to output_file.
    """
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            cleaned_data = [entry for entry in data if entry.get(key) is True]
        elif isinstance(data, dict):
            cleaned_data = {k: v for k, v in data.items() if v.get(key) is True}
        else:
            raise ValueError("Unsupported data format. Expected list or dict.")
        
        with open(output_file, 'w') as f:
            json.dump(cleaned_data, f, indent=2)
        
        print(f"Cleaned data saved to {output_file}")
        return len(cleaned_data)
    
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except json.JSONDecodeError:
        print(f"Error: File '{input_file}' contains invalid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(filepath):
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_column(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_data(df, numeric_columns):
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
        df = normalize_column(df, col)
    return df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    raw_data = load_dataset('raw_dataset.csv')
    numeric_cols = ['age', 'income', 'score']
    cleaned_df = clean_data(raw_data, numeric_cols)
    save_cleaned_data(cleaned_df, 'cleaned_dataset.csv')