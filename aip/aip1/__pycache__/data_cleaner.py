import pandas as pd
import sys

def remove_duplicates(input_file, output_file=None):
    """
    Remove duplicate rows from a CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str, optional): Path to output CSV file. 
                                    If None, overwrites input file.
    """
    try:
        df = pd.read_csv(input_file)
        initial_rows = len(df)
        df_clean = df.drop_duplicates()
        final_rows = len(df_clean)
        
        if output_file is None:
            output_file = input_file
            
        df_clean.to_csv(output_file, index=False)
        
        duplicates_removed = initial_rows - final_rows
        print(f"Successfully removed {duplicates_removed} duplicate rows.")
        print(f"Original rows: {initial_rows}, Cleaned rows: {final_rows}")
        
        return df_clean
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: File '{input_file}' is empty.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    remove_duplicates(input_file, output_file)
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(dataframe, columns):
    cleaned_df = dataframe.copy()
    for col in columns:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    return cleaned_df

def normalize_data(dataframe, columns, method='minmax'):
    normalized_df = dataframe.copy()
    for col in columns:
        if method == 'minmax':
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = normalized_df[col].mean()
            std_val = normalized_df[col].std()
            normalized_df[col] = (normalized_df[col] - mean_val) / std_val
    return normalized_df

def handle_missing_values(dataframe, columns, strategy='mean'):
    processed_df = dataframe.copy()
    for col in columns:
        if strategy == 'mean':
            fill_value = processed_df[col].mean()
        elif strategy == 'median':
            fill_value = processed_df[col].median()
        elif strategy == 'mode':
            fill_value = processed_df[col].mode()[0]
        else:
            fill_value = 0
        processed_df[col].fillna(fill_value, inplace=True)
    return processed_df

def clean_dataset(dataframe, numeric_columns):
    df_no_missing = handle_missing_values(dataframe, numeric_columns, 'median')
    df_no_outliers = remove_outliers_iqr(df_no_missing, numeric_columns)
    df_normalized = normalize_data(df_no_outliers, numeric_columns, 'zscore')
    return df_normalized