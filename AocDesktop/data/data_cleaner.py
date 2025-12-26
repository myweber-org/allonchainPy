
import pandas as pd
import sys

def remove_duplicates(input_file, output_file=None):
    """
    Read a CSV file, remove duplicate rows, and save the cleaned data.
    If no output file is specified, overwrite the input file.
    """
    try:
        df = pd.read_csv(input_file)
        initial_count = len(df)
        df_cleaned = df.drop_duplicates()
        final_count = len(df_cleaned)
        
        if output_file is None:
            output_file = input_file
        
        df_cleaned.to_csv(output_file, index=False)
        
        duplicates_removed = initial_count - final_count
        print(f"Cleaning complete. Removed {duplicates_removed} duplicate rows.")
        print(f"Original rows: {initial_count}, Cleaned rows: {final_count}")
        print(f"Saved to: {output_file}")
        
        return duplicates_removed
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File '{input_file}' is empty.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    remove_duplicates(input_file, output_file)
import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using the Interquartile Range method.
    Returns a boolean mask where True indicates an outlier.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    return (data[column] < lower_bound) | (data[column] > upper_bound)

def remove_outliers(data, columns=None, threshold=1.5):
    """
    Remove outliers from specified columns using IQR method.
    If columns is None, applies to all numeric columns.
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    outlier_mask = pd.Series(False, index=data.index)
    for col in columns:
        if col in data.columns and np.issubdtype(data[col].dtype, np.number):
            outlier_mask |= detect_outliers_iqr(data, col, threshold)
    
    return data[~outlier_mask].copy()

def normalize_minmax(data, columns=None):
    """
    Normalize data using min-max scaling to range [0, 1].
    If columns is None, applies to all numeric columns.
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    normalized_data = data.copy()
    for col in columns:
        if col in normalized_data.columns and np.issubdtype(normalized_data[col].dtype, np.number):
            col_min = normalized_data[col].min()
            col_max = normalized_data[col].max()
            if col_max != col_min:
                normalized_data[col] = (normalized_data[col] - col_min) / (col_max - col_min)
            else:
                normalized_data[col] = 0
    
    return normalized_data

def standardize_zscore(data, columns=None):
    """
    Standardize data using z-score normalization (mean=0, std=1).
    If columns is None, applies to all numeric columns.
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    standardized_data = data.copy()
    for col in columns:
        if col in standardized_data.columns and np.issubdtype(standardized_data[col].dtype, np.number):
            col_mean = standardized_data[col].mean()
            col_std = standardized_data[col].std()
            if col_std > 0:
                standardized_data[col] = (standardized_data[col] - col_mean) / col_std
            else:
                standardized_data[col] = 0
    
    return standardized_data

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy.
    Strategies: 'mean', 'median', 'mode', 'drop', 'constant'
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    processed_data = data.copy()
    
    for col in columns:
        if col not in processed_data.columns:
            continue
            
        if strategy == 'drop':
            processed_data = processed_data.dropna(subset=[col])
        elif strategy == 'mean' and np.issubdtype(processed_data[col].dtype, np.number):
            processed_data[col] = processed_data[col].fillna(processed_data[col].mean())
        elif strategy == 'median' and np.issubdtype(processed_data[col].dtype, np.number):
            processed_data[col] = processed_data[col].fillna(processed_data[col].median())
        elif strategy == 'mode':
            mode_value = processed_data[col].mode()
            if not mode_value.empty:
                processed_data[col] = processed_data[col].fillna(mode_value[0])
        elif strategy == 'constant':
            processed_data[col] = processed_data[col].fillna(0)
    
    return processed_data

def clean_dataset(data, numeric_columns=None, outlier_threshold=1.5, 
                  normalize_method='standardize', missing_strategy='mean'):
    """
    Comprehensive data cleaning pipeline.
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    cleaned_data = handle_missing_values(cleaned_data, strategy=missing_strategy, columns=numeric_columns)
    cleaned_data = remove_outliers(cleaned_data, columns=numeric_columns, threshold=outlier_threshold)
    
    if normalize_method == 'minmax':
        cleaned_data = normalize_minmax(cleaned_data, columns=numeric_columns)
    elif normalize_method == 'standardize':
        cleaned_data = standardize_zscore(cleaned_data, columns=numeric_columns)
    
    return cleaned_data