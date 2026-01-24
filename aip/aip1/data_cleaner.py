
import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using IQR method.
    Returns boolean mask for outliers.
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return (data[column] < lower_bound) | (data[column] > upper_bound)

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    Returns filtered DataFrame.
    """
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    """
    Normalize column using Min-Max scaling.
    Returns normalized Series.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    return (data[column] - min_val) / (max_val - min_val)

def normalize_standard(data, column):
    """
    Normalize column using Standard scaling.
    Returns normalized Series.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    return (data[column] - mean_val) / std_val

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='standard'):
    """
    Main cleaning function that handles outliers and normalization.
    Returns cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col not in cleaned_df.columns:
            continue
            
        if outlier_method == 'iqr':
            outliers = detect_outliers_iqr(cleaned_df, col)
            cleaned_df = cleaned_df[~outliers]
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        
        if normalize_method == 'minmax':
            cleaned_df[f'{col}_normalized'] = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'standard':
            cleaned_df[f'{col}_normalized'] = normalize_standard(cleaned_df, col)
    
    return cleaned_df

def get_summary_statistics(df, numeric_columns):
    """
    Generate summary statistics for numeric columns.
    Returns dictionary with statistics.
    """
    summary = {}
    for col in numeric_columns:
        if col in df.columns:
            summary[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'count': df[col].count(),
                'missing': df[col].isnull().sum()
            }
    return summary

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    cleaned = clean_dataset(
        sample_data, 
        ['feature1', 'feature2'], 
        outlier_method='iqr', 
        normalize_method='standard'
    )
    
    stats_summary = get_summary_statistics(cleaned, ['feature1', 'feature2'])
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned.shape}")
    print(f"Removed rows: {len(sample_data) - len(cleaned)}")import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
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

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): List of column names to clean
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    cleaned_df = df.copy()
    for col in columns:
        if col in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[col]):
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    return cleaned_df

def get_statistics(df, column):
    """
    Get statistical summary for a column including outlier bounds.
    
    Args:
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
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    stats['q1'] = Q1
    stats['q3'] = Q3
    stats['iqr'] = IQR
    stats['lower_bound'] = Q1 - 1.5 * IQR
    stats['upper_bound'] = Q3 + 1.5 * IQR
    
    return stats

if __name__ == "__main__":
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 27, 28, 29, 30, 100],
        'humidity': [45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
        'pressure': [1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 950]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nStatistics for 'temperature':")
    print(get_statistics(df, 'temperature'))
    
    cleaned_df = clean_numeric_data(df, ['temperature', 'pressure'])
    print("\nCleaned DataFrame:")
    print(cleaned_df)