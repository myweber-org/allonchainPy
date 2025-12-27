
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def calculate_summary_statistics(df):
    """
    Calculate summary statistics for numeric columns in DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    pd.DataFrame: Summary statistics
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary = df[numeric_cols].agg(['count', 'mean', 'std', 'min', 'max'])
    return summary

def clean_missing_values(df, strategy='mean'):
    """
    Handle missing values in numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Strategy for handling missing values ('mean', 'median', 'drop')
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        df_cleaned = df.copy()
        for col in numeric_cols:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
    elif strategy == 'median':
        df_cleaned = df.copy()
        for col in numeric_cols:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
    elif strategy == 'drop':
        df_cleaned = df.dropna(subset=numeric_cols)
    else:
        raise ValueError("Strategy must be 'mean', 'median', or 'drop'")
    
    return df_cleaned

def normalize_data(df, columns=None):
    """
    Normalize specified columns using min-max scaling.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to normalize
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    df_normalized = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns and np.issubdtype(df[col].dtype, np.number):
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_max != col_min:
                df_normalized[col] = (df[col] - col_min) / (col_max - col_min)
            else:
                df_normalized[col] = 0
    
    return df_normalized

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 3, 4, 5, 100],
        'B': [10, 20, 30, 40, 50, 60],
        'C': [100, 200, 300, 400, 500, 600]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned_df = remove_outliers_iqr(df, 'A')
    print("DataFrame after removing outliers from column 'A':")
    print(cleaned_df)
    print()
    
    stats = calculate_summary_statistics(df)
    print("Summary statistics:")
    print(stats)
    print()
    
    normalized_df = normalize_data(df, ['A', 'B', 'C'])
    print("Normalized DataFrame:")
    print(normalized_df)