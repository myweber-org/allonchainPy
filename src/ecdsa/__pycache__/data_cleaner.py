import pandas as pd
import numpy as np

def detect_outliers_iqr(data, column):
    """
    Detect outliers using the Interquartile Range (IQR) method.
    Returns a boolean Series where True indicates an outlier.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data[column] < lower_bound) | (data[column] > upper_bound)

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns using the given strategy.
    Supported strategies: 'mean', 'median', 'mode', 'drop'.
    """
    if columns is None:
        columns = data.columns
    
    data_filled = data.copy()
    
    for col in columns:
        if data[col].isnull().any():
            if strategy == 'mean':
                fill_value = data[col].mean()
            elif strategy == 'median':
                fill_value = data[col].median()
            elif strategy == 'mode':
                fill_value = data[col].mode()[0]
            elif strategy == 'drop':
                data_filled = data_filled.dropna(subset=[col])
                continue
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")
            
            data_filled[col] = data_filled[col].fillna(fill_value)
    
    return data_filled

def remove_duplicates(data, subset=None, keep='first'):
    """
    Remove duplicate rows from the dataset.
    """
    return data.drop_duplicates(subset=subset, keep=keep)

def normalize_column(data, column, method='minmax'):
    """
    Normalize a column using specified method.
    Supported methods: 'minmax', 'zscore'.
    """
    if method == 'minmax':
        min_val = data[column].min()
        max_val = data[column].max()
        if max_val != min_val:
            data[column] = (data[column] - min_val) / (max_val - min_val)
    elif method == 'zscore':
        mean_val = data[column].mean()
        std_val = data[column].std()
        if std_val != 0:
            data[column] = (data[column] - mean_val) / std_val
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    
    return data

def clean_dataset(data, missing_strategy='mean', normalize_columns=None, remove_outliers=False):
    """
    Comprehensive data cleaning pipeline.
    """
    cleaned_data = data.copy()
    
    cleaned_data = handle_missing_values(cleaned_data, strategy=missing_strategy)
    cleaned_data = remove_duplicates(cleaned_data)
    
    if remove_outliers:
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            outliers = detect_outliers_iqr(cleaned_data, col)
            cleaned_data = cleaned_data[~outliers]
    
    if normalize_columns:
        for col in normalize_columns:
            if col in cleaned_data.columns:
                cleaned_data = normalize_column(cleaned_data, col, method='minmax')
    
    return cleaned_data

def validate_data(data, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    """
    if len(data) < min_rows:
        raise ValueError(f"Dataset must have at least {min_rows} rows")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True
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

def calculate_summary_stats(df, column):
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

def clean_dataset(df, columns_to_clean=None):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary of statistics for each cleaned column
    """
    if columns_to_clean is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns_to_clean = list(numeric_cols)
    
    cleaned_df = df.copy()
    stats_dict = {}
    
    for column in columns_to_clean:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            stats = calculate_summary_stats(cleaned_df, column)
            stats['outliers_removed'] = removed_count
            stats_dict[column] = stats
    
    return cleaned_df, stats_dict

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[::100, 'A'] = 500
    
    print("Original DataFrame shape:", df.shape)
    print("\nOriginal summary statistics:")
    print(df.describe())
    
    cleaned_df, stats = clean_dataset(df, ['A', 'B'])
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("\nCleaning statistics:")
    for col, col_stats in stats.items():
        print(f"\n{col}:")
        for stat_name, stat_value in col_stats.items():
            print(f"  {stat_name}: {stat_value:.2f}")