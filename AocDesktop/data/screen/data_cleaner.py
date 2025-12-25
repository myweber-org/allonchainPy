import pandas as pd
import argparse
import sys

def remove_duplicates(input_file, output_file, subset=None, keep='first'):
    """
    Remove duplicate rows from a CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
        subset (list): Columns to consider for identifying duplicates
        keep (str): Which duplicates to keep - 'first', 'last', or False
    """
    try:
        df = pd.read_csv(input_file)
        initial_count = len(df)
        
        df_cleaned = df.drop_duplicates(subset=subset, keep=keep)
        final_count = len(df_cleaned)
        
        df_cleaned.to_csv(output_file, index=False)
        
        duplicates_removed = initial_count - final_count
        print(f"Successfully removed {duplicates_removed} duplicate rows")
        print(f"Original rows: {initial_count}, Cleaned rows: {final_count}")
        print(f"Cleaned data saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: Input file '{input_file}' is empty")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Remove duplicate rows from CSV files')
    parser.add_argument('input', help='Input CSV file path')
    parser.add_argument('output', help='Output CSV file path')
    parser.add_argument('--subset', nargs='+', help='Columns to consider for duplicates')
    parser.add_argument('--keep', choices=['first', 'last', 'none'], 
                       default='first', help='Which duplicates to keep')
    
    args = parser.parse_args()
    
    keep_value = False if args.keep == 'none' else args.keep
    remove_duplicates(args.input, args.output, args.subset, keep_value)

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, columns=None, factor=1.5):
    """
    Remove outliers from specified columns using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to process. If None, process all numeric columns.
    factor (float): Multiplier for IQR. Default is 1.5.
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize specified columns in dataframe.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to normalize
    method (str): Normalization method: 'minmax' or 'zscore'
    
    Returns:
    pd.DataFrame: Dataframe with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_norm = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val != min_val:
                df_norm[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df_norm[col] = 0
        
        elif method == 'zscore':
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val != 0:
                df_norm[col] = (df[col] - mean_val) / std_val
            else:
                df_norm[col] = 0
    
    return df_norm

def handle_missing_values(df, columns=None, strategy='mean'):
    """
    Handle missing values in specified columns.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to process
    strategy (str): Strategy for imputation: 'mean', 'median', 'mode', or 'drop'
    
    Returns:
    pd.DataFrame: Dataframe with handled missing values
    """
    if columns is None:
        columns = df.columns.tolist()
    
    df_processed = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if df[col].isnull().any():
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
            elif strategy == 'drop':
                df_processed = df_processed.dropna(subset=[col])
                continue
            
            df_processed[col] = df[col].fillna(fill_value)
    
    return df_processed.reset_index(drop=True)