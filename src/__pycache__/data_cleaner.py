
def remove_duplicates(input_list):
    """
    Remove duplicate elements from a list while preserving order.
    
    Args:
        input_list (list): The list from which duplicates are to be removed.
    
    Returns:
        list: A new list with duplicates removed.
    """
    seen = set()
    result = []
    
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    
    return result

def clean_data_with_threshold(data, threshold=None):
    """
    Clean data by removing duplicates, optionally with a frequency threshold.
    
    Args:
        data (list): Input data list.
        threshold (int, optional): Minimum frequency to keep an item. Defaults to None.
    
    Returns:
        list: Cleaned data list.
    """
    cleaned = remove_duplicates(data)
    
    if threshold is not None and threshold > 0:
        from collections import Counter
        counts = Counter(data)
        cleaned = [item for item in cleaned if counts[item] >= threshold]
    
    return cleaned

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, 4, 4, 4, 5, 1, 6]
    print("Original data:", sample_data)
    print("Without duplicates:", remove_duplicates(sample_data))
    print("With threshold 2:", clean_data_with_threshold(sample_data, threshold=2))
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
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
    
    return filtered_df.reset_index(drop=True)

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': len(df[column])
    }
    
    return stats

def clean_dataset(df, columns_to_clean):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary of statistics for each cleaned column
    """
    cleaned_df = df.copy()
    all_stats = {}
    
    for column in columns_to_clean:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            
            stats = calculate_statistics(cleaned_df, column)
            stats['outliers_removed'] = removed_count
            all_stats[column] = stats
    
    return cleaned_df, all_stats

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    sample_data = {
        'temperature': np.random.normal(25, 5, 1000),
        'humidity': np.random.normal(60, 15, 1000),
        'pressure': np.random.normal(1013, 10, 1000)
    }
    
    # Add some outliers
    sample_df = pd.DataFrame(sample_data)
    sample_df.loc[1000] = [100, 200, 500]  # Extreme outlier
    sample_df.loc[1001] = [-50, -30, 800]  # Negative outlier
    
    print("Original dataset shape:", sample_df.shape)
    
    columns_to_process = ['temperature', 'humidity', 'pressure']
    cleaned_df, statistics = clean_dataset(sample_df, columns_to_process)
    
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("\nStatistics for cleaned columns:")
    
    for col, stats in statistics.items():
        print(f"\n{col}:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")