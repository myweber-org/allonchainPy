
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to clean.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
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

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing statistical measures.
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

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    data = pd.DataFrame({
        'values': np.random.normal(100, 15, 1000)
    })
    
    print("Original data shape:", data.shape)
    print("Original statistics:", calculate_basic_stats(data, 'values'))
    
    cleaned_data = remove_outliers_iqr(data, 'values')
    
    print("\nCleaned data shape:", cleaned_data.shape)
    print("Cleaned statistics:", calculate_basic_stats(cleaned_data, 'values'))

if __name__ == "__main__":
    example_usage()import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    """
    Remove outliers using the Interquartile Range method.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    """
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(df, numeric_columns, method='iqr', normalize=False):
    """
    Clean dataset by removing outliers and optionally normalizing numeric columns.
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        
        if normalize:
            cleaned_df = normalize_minmax(cleaned_df, col)
    
    return cleaned_df.reset_index(drop=True)

def get_summary_statistics(df, numeric_columns):
    """
    Calculate summary statistics for numeric columns.
    """
    summary = {}
    for col in numeric_columns:
        summary[col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'count': df[col].count(),
            'missing': df[col].isnull().sum()
        }
    return pd.DataFrame(summary).T

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[np.random.choice(df.index, 50), 'A'] = np.nan
    
    print("Original dataset shape:", df.shape)
    print("\nOriginal summary statistics:")
    print(get_summary_statistics(df, ['A', 'B', 'C']))
    
    cleaned_df = clean_dataset(df, ['A', 'B', 'C'], method='iqr', normalize=True)
    
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("\nCleaned summary statistics:")
    print(get_summary_statistics(cleaned_df, ['A', 'B', 'C']))
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    return dataframe[(dataframe[column] >= lower_bound) & 
                     (dataframe[column] <= upper_bound)]

def zscore_normalize(dataframe, column):
    """
    Normalize column using z-score normalization
    """
    mean = dataframe[column].mean()
    std = dataframe[column].std()
    
    if std > 0:
        dataframe[column] = (dataframe[column] - mean) / std
    return dataframe

def minmax_normalize(dataframe, column):
    """
    Normalize column using min-max scaling
    """
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()
    
    if max_val > min_val:
        dataframe[column] = (dataframe[column] - min_val) / (max_val - min_val)
    return dataframe

def detect_skewed_columns(dataframe, threshold=0.5):
    """
    Detect columns with significant skewness
    """
    skewed_cols = []
    for col in dataframe.select_dtypes(include=[np.number]).columns:
        skewness = dataframe[col].skew()
        if abs(skewness) > threshold:
            skewed_cols.append((col, skewness))
    
    return sorted(skewed_cols, key=lambda x: abs(x[1]), reverse=True)

def apply_log_transform(dataframe, column):
    """
    Apply log transformation to reduce skewness
    """
    if dataframe[column].min() <= 0:
        dataframe[column] = dataframe[column] - dataframe[column].min() + 1
    
    dataframe[column] = np.log(dataframe[column])
    return dataframe

def clean_dataset(dataframe, numeric_columns=None, outlier_threshold=1.5, 
                  normalization_method='zscore', skew_threshold=0.5):
    """
    Main cleaning pipeline
    """
    df_clean = dataframe.copy()
    
    if numeric_columns is None:
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean = remove_outliers_iqr(df_clean, col, outlier_threshold)
            
            if normalization_method == 'zscore':
                df_clean = zscore_normalize(df_clean, col)
            elif normalization_method == 'minmax':
                df_clean = minmax_normalize(df_clean, col)
    
    skewed_cols = detect_skewed_columns(df_clean, skew_threshold)
    for col, _ in skewed_cols:
        if col in df_clean.columns:
            df_clean = apply_log_transform(df_clean, col)
    
    return df_clean, skewed_cols