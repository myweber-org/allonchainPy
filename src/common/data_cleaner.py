
def remove_duplicates(data_list):
    """
    Remove duplicate entries from a list while preserving order.
    
    Args:
        data_list: List containing potentially duplicate items.
    
    Returns:
        List with duplicates removed.
    """
    seen = set()
    result = []
    
    for item in data_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    
    return result

def clean_numeric_data(values, remove_negative=False):
    """
    Clean numeric data by removing non-numeric values and optionally negative values.
    
    Args:
        values: List of values to clean.
        remove_negative: Boolean flag to remove negative values.
    
    Returns:
        List of cleaned numeric values.
    """
    cleaned = []
    
    for value in values:
        try:
            num = float(value)
            if remove_negative and num < 0:
                continue
            cleaned.append(num)
        except (ValueError, TypeError):
            continue
    
    return cleaned

if __name__ == "__main__":
    # Example usage
    sample_data = [1, 2, 2, 3, 4, 4, 5, 1, 6]
    cleaned_data = remove_duplicates(sample_data)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {cleaned_data}")
    
    mixed_data = [1, "2", "abc", 3.5, -2, None, 7]
    numeric_data = clean_numeric_data(mixed_data, remove_negative=True)
    print(f"Mixed data: {mixed_data}")
    print(f"Numeric only: {numeric_data}")
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
    
    return filtered_df

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
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
    dict: Dictionary of summary statistics for each cleaned column
    """
    cleaned_df = df.copy()
    all_stats = {}
    
    for column in columns_to_clean:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            
            stats = calculate_summary_statistics(cleaned_df, column)
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
    
    sample_df = pd.DataFrame(sample_data)
    
    # Add some outliers
    sample_df.loc[1000] = [150, 60, 1013]  # Extreme temperature
    sample_df.loc[1001] = [25, 200, 1013]  # Extreme humidity
    
    print("Original dataset shape:", sample_df.shape)
    
    columns_to_clean = ['temperature', 'humidity']
    cleaned_df, stats = clean_dataset(sample_df, columns_to_clean)
    
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("\nSummary statistics:")
    for col, col_stats in stats.items():
        print(f"\n{col}:")
        for stat_name, stat_value in col_stats.items():
            print(f"  {stat_name}: {stat_value:.2f}")