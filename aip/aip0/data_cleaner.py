
def remove_duplicates(input_list):
    """
    Remove duplicate elements from a list while preserving order.
    
    Args:
        input_list: A list that may contain duplicate elements.
    
    Returns:
        A new list with duplicates removed.
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
    Clean data by removing duplicates, optionally filtering by frequency threshold.
    
    Args:
        data: List of data items.
        threshold: Minimum frequency count to keep an item (inclusive).
    
    Returns:
        Cleaned list of items.
    """
    from collections import Counter
    
    if not data:
        return []
    
    counter = Counter(data)
    
    if threshold is None:
        return remove_duplicates(data)
    
    cleaned = [item for item in data if counter[item] >= threshold]
    return remove_duplicates(cleaned)

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, 4, 4, 4, 5, 1]
    print(f"Original data: {sample_data}")
    print(f"Without duplicates: {remove_duplicates(sample_data)}")
    print(f"Threshold >= 2: {clean_data_with_threshold(sample_data, threshold=2)}")
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
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
    Calculate basic statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name
    
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
        'count': len(df[column]),
        'non_null_count': df[column].count()
    }
    
    return stats

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100)
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    print("Original statistics:", calculate_statistics(df, 'value'))
    
    cleaned_df = remove_outliers_iqr(df, 'value')
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned statistics:", calculate_statistics(cleaned_df, 'value'))
    
    return cleaned_df

if __name__ == "__main__":
    result_df = example_usage()
    print(f"\nRemoved {100 - len(result_df)} outliers")