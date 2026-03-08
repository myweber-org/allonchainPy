import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers from a pandas Series using the IQR method.
    
    Parameters:
    data (pd.Series): Input data series
    column (str): Column name to process
    factor (float): Multiplier for IQR (default 1.5)
    
    Returns:
    pd.DataFrame: Data with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    series = data[column]
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(series >= lower_bound) & (series <= upper_bound)]
    return filtered_data.copy()

def normalize_minmax(data, column):
    """
    Normalize data to [0, 1] range using min-max scaling.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to normalize
    
    Returns:
    pd.Series: Normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    series = data[column]
    min_val = series.min()
    max_val = series.max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(series), index=series.index)
    
    normalized = (series - min_val) / (max_val - min_val)
    return normalized

def clean_dataset(df, numeric_columns=None, outlier_factor=1.5):
    """
    Clean dataset by removing outliers and normalizing numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    numeric_columns (list): List of numeric column names to process
    outlier_factor (float): Factor for IQR outlier detection
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in df.columns:
            # Remove outliers
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_factor)
            
            # Normalize the column
            cleaned_df[f"{col}_normalized"] = normalize_minmax(cleaned_df, col)
    
    return cleaned_df.reset_index(drop=True)

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    column (str): Column name
    
    Returns:
    dict: Dictionary of statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    series = df[column]
    stats = {
        'mean': series.mean(),
        'median': series.median(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max(),
        'count': series.count(),
        'missing': series.isnull().sum()
    }
    
    return stats

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'temperature': [22.5, 23.1, 21.8, 100.2, 22.9, 23.5, -10.3, 22.7],
        'humidity': [45, 47, 43, 46, 48, 200, 44, 46],
        'label': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    })
    
    print("Original data:")
    print(sample_data)
    print("\nStatistics for temperature:")
    print(calculate_statistics(sample_data, 'temperature'))
    
    cleaned = clean_dataset(sample_data, ['temperature', 'humidity'])
    print("\nCleaned data:")
    print(cleaned)import pandas as pd
import sys

def remove_duplicates(input_file, output_file=None, subset=None, keep='first'):
    """
    Remove duplicate rows from a CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str, optional): Path to output CSV file. If None, overwrites input file
        subset (list, optional): Columns to consider for identifying duplicates
        keep (str): Which duplicates to keep - 'first', 'last', or False to drop all
        
    Returns:
        int: Number of duplicate rows removed
    """
    try:
        df = pd.read_csv(input_file)
        initial_rows = len(df)
        
        df_clean = df.drop_duplicates(subset=subset, keep=keep)
        final_rows = len(df_clean)
        
        duplicates_removed = initial_rows - final_rows
        
        if output_file is None:
            output_file = input_file
            
        df_clean.to_csv(output_file, index=False)
        
        print(f"Removed {duplicates_removed} duplicate rows")
        print(f"Original rows: {initial_rows}, Cleaned rows: {final_rows}")
        print(f"Saved to: {output_file}")
        
        return duplicates_removed
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        return -1
    except Exception as e:
        print(f"Error processing file: {e}")
        return -1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = remove_duplicates(input_file, output_file)
    
    if result >= 0:
        sys.exit(0)
    else:
        sys.exit(1)