
import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: pandas DataFrame to clean
        drop_duplicates: Boolean indicating whether to remove duplicate rows
        fill_missing: Method to fill missing values ('mean', 'median', 'mode', or a specific value)
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        if fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
        else:
            cleaned_df = cleaned_df.fillna(fill_missing)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate a DataFrame for basic integrity checks.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of column names that must be present
        min_rows: Minimum number of rows required
    
    Returns:
        Boolean indicating whether DataFrame passes validation
    """
    if df.empty:
        return False
    
    if len(df) < min_rows:
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False
    
    return True

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a DataFrame column using specified method.
    
    Args:
        df: pandas DataFrame
        column: Column name to process
        method: Outlier detection method ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in df.columns:
        return df
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    elif method == 'zscore':
        from scipy import stats
        z_scores = stats.zscore(df[column])
        filtered_df = df[(abs(z_scores) < threshold)]
    else:
        filtered_df = df
    
    return filtered_df
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    data (list or np.array): The dataset containing the column to clean.
    column (int): Index of the column to process.
    
    Returns:
    np.array: Data with outliers removed from the specified column.
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    column_data = data[:, column].astype(float)
    
    Q1 = np.percentile(column_data, 25)
    Q3 = np.percentile(column_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (column_data >= lower_bound) & (column_data <= upper_bound)
    
    return data[mask]

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    data (np.array): The cleaned dataset.
    column (int): Index of the column to analyze.
    
    Returns:
    dict: Dictionary containing mean, median, and standard deviation.
    """
    column_data = data[:, column].astype(float)
    
    stats = {
        'mean': np.mean(column_data),
        'median': np.median(column_data),
        'std_dev': np.std(column_data),
        'count': len(column_data)
    }
    
    return stats

def clean_dataset(data, columns_to_clean):
    """
    Clean multiple columns in a dataset by removing outliers.
    
    Parameters:
    data (list or np.array): The original dataset.
    columns_to_clean (list): List of column indices to clean.
    
    Returns:
    tuple: (cleaned_data, removal_stats) where removal_stats is a dictionary
           showing how many rows were removed for each column.
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    cleaned_data = data.copy()
    removal_stats = {}
    
    for column in columns_to_clean:
        original_count = len(cleaned_data)
        cleaned_data = remove_outliers_iqr(cleaned_data, column)
        removed_count = original_count - len(cleaned_data)
        removal_stats[column] = removed_count
    
    return cleaned_data, removal_stats

if __name__ == "__main__":
    # Example usage
    sample_data = np.array([
        [1, 150, 25],
        [2, 160, 30],
        [3, 170, 35],
        [4, 180, 40],
        [5, 190, 45],
        [6, 1000, 200],  # Outlier
        [7, 200, 50],
        [8, 210, 55],
        [9, 220, 60],
        [10, 230, 65]
    ])
    
    print("Original data shape:", sample_data.shape)
    
    cleaned_data, stats = clean_dataset(sample_data, [1, 2])
    
    print("Cleaned data shape:", cleaned_data.shape)
    print("Rows removed per column:", stats)
    
    for col in [1, 2]:
        col_stats = calculate_statistics(cleaned_data, col)
        print(f"\nStatistics for column {col}:")
        for key, value in col_stats.items():
            print(f"  {key}: {value:.2f}")