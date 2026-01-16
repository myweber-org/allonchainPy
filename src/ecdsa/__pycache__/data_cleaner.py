import pandas as pd
import numpy as np

def clean_csv_data(filepath, missing_strategy='mean', columns_to_drop=None):
    """
    Load and clean CSV data by handling missing values and removing specified columns.
    
    Args:
        filepath (str): Path to the CSV file
        missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'drop', 'zero')
        columns_to_drop (list): List of column names to remove from dataset
    
    Returns:
        pandas.DataFrame: Cleaned dataframe
    """
    try:
        df = pd.read_csv(filepath)
        
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop, errors='ignore')
        
        if missing_strategy == 'mean':
            df = df.fillna(df.mean(numeric_only=True))
        elif missing_strategy == 'median':
            df = df.fillna(df.median(numeric_only=True))
        elif missing_strategy == 'zero':
            df = df.fillna(0)
        elif missing_strategy == 'drop':
            df = df.dropna()
        else:
            raise ValueError(f"Unsupported missing strategy: {missing_strategy}")
        
        df = df.reset_index(drop=True)
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def detect_outliers_iqr(df, column, threshold=1.5):
    """
    Detect outliers using the Interquartile Range method.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        column (str): Column name to analyze
        threshold (float): IQR multiplier for outlier detection
    
    Returns:
        tuple: (lower_bound, upper_bound, outlier_indices)
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    return lower_bound, upper_bound, outliers.index.tolist()

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        column (str): Column name to normalize
        method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
        pandas.DataFrame: Dataframe with normalized column
    """
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        if max_val != min_val:
            df_copy[f'{column}_normalized'] = (df_copy[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val > 0:
            df_copy[f'{column}_normalized'] = (df_copy[column] - mean_val) / std_val
    
    return df_copy

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, 20, 30, np.nan, 50],
        'C': [100, 200, 300, 400, 500]
    })
    
    cleaned = clean_csv_data('sample.csv', missing_strategy='mean')
    print("Data cleaning completed")