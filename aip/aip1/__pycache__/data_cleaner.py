
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
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

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary of removal statistics for each column
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    original_shape = df.shape
    removal_stats = {}
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            original_count = cleaned_df.shape[0]
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_count - cleaned_df.shape[0]
            removal_stats[col] = removed_count
    
    final_shape = cleaned_df.shape
    removal_stats['total_removed'] = original_shape[0] - final_shape[0]
    removal_stats['remaining_percentage'] = (final_shape[0] / original_shape[0]) * 100
    
    return cleaned_df, removal_stats

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[np.random.choice(df.index, 50), 'A'] = np.random.uniform(500, 1000, 50)
    
    print("Original DataFrame shape:", df.shape)
    print("\nOriginal statistics for column 'A':")
    print(calculate_basic_stats(df, 'A'))
    
    cleaned_df, stats = clean_numeric_data(df, ['A', 'B'])
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("\nRemoval statistics:", stats)
    print("\nCleaned statistics for column 'A':")
    print(calculate_basic_stats(cleaned_df, 'A'))