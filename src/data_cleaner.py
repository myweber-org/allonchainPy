
import pandas as pd
import numpy as np

def clean_csv_data(file_path, fill_method='mean', drop_threshold=0.5):
    """
    Load and clean CSV data by handling missing values.
    
    Parameters:
    file_path (str): Path to the CSV file.
    fill_method (str): Method to fill missing values ('mean', 'median', 'mode', 'zero').
    drop_threshold (float): Drop columns with missing ratio above this threshold.
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    missing_ratio = df.isnull().sum() / len(df)
    columns_to_drop = missing_ratio[missing_ratio > drop_threshold].index
    df = df.drop(columns=columns_to_drop)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    
    if fill_method == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif fill_method == 'median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif fill_method == 'zero':
        df[numeric_cols] = df[numeric_cols].fillna(0)
    
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')
    
    return df

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a numeric column using IQR method.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame.
    column (str): Column name to process.
    
    Returns:
    pandas.DataFrame: DataFrame with outliers removed.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def standardize_columns(df, columns):
    """
    Standardize specified numeric columns to zero mean and unit variance.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame.
    columns (list): List of column names to standardize.
    
    Returns:
    pandas.DataFrame: DataFrame with standardized columns.
    """
    df_standardized = df.copy()
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df_standardized[col] = (df[col] - mean) / std
    return df_standardized

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, np.nan, np.nan, 10, 11],
        'C': ['x', 'y', np.nan, 'z', 'w'],
        'D': [100, 200, 300, 400, 500]
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned = clean_csv_data('sample_data.csv', fill_method='median', drop_threshold=0.6)
    print("Cleaned DataFrame:")
    print(cleaned)
    
    cleaned_no_outliers = remove_outliers_iqr(cleaned, 'D')
    print("\nDataFrame after outlier removal:")
    print(cleaned_no_outliers)
    
    standardized = standardize_columns(cleaned_no_outliers, ['D'])
    print("\nDataFrame after standardization:")
    print(standardized)import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    """
    Main function to clean dataset by removing outliers and normalizing numeric columns.
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col not in cleaned_df.columns:
            print(f"Warning: Column '{col}' not found, skipping...")
            continue
        
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        else:
            raise ValueError("Invalid outlier_method. Choose 'iqr' or 'zscore'")
        
        if normalize_method == 'minmax':
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df[col] = normalize_zscore(cleaned_df, col)
        else:
            raise ValueError("Invalid normalize_method. Choose 'minmax' or 'zscore'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    print("Original dataset shape:", sample_data.shape)
    print("Original statistics:")
    print(sample_data[['feature1', 'feature2']].describe())
    
    cleaned_data = clean_dataset(
        sample_data, 
        numeric_columns=['feature1', 'feature2'],
        outlier_method='iqr',
        normalize_method='zscore'
    )
    
    print("\nCleaned dataset shape:", cleaned_data.shape)
    print("Cleaned statistics:")
    print(cleaned_data[['feature1', 'feature2']].describe())