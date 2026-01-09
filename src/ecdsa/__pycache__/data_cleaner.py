
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
    
    return filtered_df.reset_index(drop=True)

def detect_outliers_iqr(df, column):
    """
    Detect outliers in a DataFrame column using IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    pd.Series: Boolean series indicating outliers (True = outlier)
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    return outliers

def calculate_outlier_statistics(df, column):
    """
    Calculate outlier statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing outlier statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    outliers = detect_outliers_iqr(df, column)
    total_count = len(df)
    outlier_count = outliers.sum()
    
    stats = {
        'total_samples': total_count,
        'outlier_count': int(outlier_count),
        'outlier_percentage': round((outlier_count / total_count) * 100, 2),
        'min_value': float(df[column].min()),
        'max_value': float(df[column].max()),
        'mean': float(df[column].mean()),
        'median': float(df[column].median()),
        'std_dev': float(df[column].std())
    }
    
    return stats

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    data = pd.DataFrame({
        'values': np.concatenate([
            np.random.normal(100, 15, 95),
            np.random.normal(300, 50, 5)
        ])
    })
    
    print("Original data shape:", data.shape)
    print("\nOutlier statistics:")
    stats = calculate_outlier_statistics(data, 'values')
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    cleaned_data = remove_outliers_iqr(data, 'values')
    print("\nCleaned data shape:", cleaned_data.shape)