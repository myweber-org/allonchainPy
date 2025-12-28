import pandas as pd
import numpy as np

def remove_duplicates(df):
    """Remove duplicate rows from DataFrame."""
    return df.drop_duplicates()

def fill_missing_values(df, strategy='mean'):
    """Fill missing values using specified strategy."""
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    else:
        return df.fillna(0)

def normalize_column(df, column_name):
    """Normalize specified column to range [0,1]."""
    if column_name in df.columns:
        col = df[column_name]
        df[column_name] = (col - col.min()) / (col.max() - col.min())
    return df

def remove_outliers(df, column_name, threshold=3):
    """Remove outliers using z-score method."""
    if column_name in df.columns:
        z_scores = np.abs((df[column_name] - df[column_name].mean()) / df[column_name].std())
        return df[z_scores < threshold]
    return df

def clean_dataframe(df, operations=None):
    """Apply multiple cleaning operations to DataFrame."""
    if operations is None:
        operations = ['remove_duplicates', 'fill_missing']
    
    cleaned_df = df.copy()
    
    if 'remove_duplicates' in operations:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if 'fill_missing' in operations:
        cleaned_df = fill_missing_values(cleaned_df)
    
    return cleaned_df
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
        'count': len(df[column])
    }
    
    return stats

def process_dataset(file_path, column_name):
    """
    Main function to load and process a dataset.
    
    Parameters:
    file_path (str): Path to CSV file
    column_name (str): Column to clean
    
    Returns:
    tuple: (cleaned_df, original_stats, cleaned_stats)
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    original_stats = calculate_summary_statistics(df, column_name)
    cleaned_df = remove_outliers_iqr(df, column_name)
    cleaned_stats = calculate_summary_statistics(cleaned_df, column_name)
    
    return cleaned_df, original_stats, cleaned_stats

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'values': np.concatenate([
            np.random.normal(100, 10, 95),
            np.random.normal(300, 10, 5)
        ])
    })
    
    print("Original data shape:", sample_data.shape)
    cleaned_data = remove_outliers_iqr(sample_data, 'values')
    print("Cleaned data shape:", cleaned_data.shape)
    
    original_stats = calculate_summary_statistics(sample_data, 'values')
    cleaned_stats = calculate_summary_statistics(cleaned_data, 'values')
    
    print("\nOriginal statistics:", original_stats)
    print("Cleaned statistics:", cleaned_stats)