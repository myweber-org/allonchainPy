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
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(df, numeric_columns):
    """
    Clean a dataset by removing outliers from multiple numeric columns.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    numeric_columns (list): List of numeric column names to clean.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    df = pd.DataFrame(data)
    
    print("Original shape:", df.shape)
    cleaned = clean_dataset(df, ['A', 'B', 'C'])
    print("Cleaned shape:", cleaned.shape)
    print("Outliers removed:", len(df) - len(cleaned))
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
    Calculate summary statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name
    
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
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def clean_numeric_data(df, columns=None):
    """
    Clean numeric columns by removing outliers and filling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for column in columns:
        if column in cleaned_df.columns:
            # Fill missing values with median
            median_value = cleaned_df[column].median()
            cleaned_df[column] = cleaned_df[column].fillna(median_value)
            
            # Remove outliers
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
    
    return cleaned_df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, 3, 4, 5, 100, 6, 7, 8, 9, 10],
        'B': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
        'C': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nSummary statistics for column 'A':")
    print(calculate_summary_stats(df, 'A'))
    
    cleaned_df = clean_numeric_data(df, columns=['A', 'B', 'C'])
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nSummary statistics for column 'A' after cleaning:")
    print(calculate_summary_stats(cleaned_df, 'A'))import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns):
    cleaned_df = df.copy()
    for col in columns:
        if col in cleaned_df.columns:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    return cleaned_df.reset_index(drop=True)

def normalize_minmax(df, columns):
    normalized_df = df.copy()
    for col in columns:
        if col in normalized_df.columns:
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            if max_val != min_val:
                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
    return normalized_df

def standardize_zscore(df, columns):
    standardized_df = df.copy()
    for col in columns:
        if col in standardized_df.columns:
            mean_val = standardized_df[col].mean()
            std_val = standardized_df[col].std()
            if std_val > 0:
                standardized_df[col] = (standardized_df[col] - mean_val) / std_val
    return standardized_df

def handle_missing_values(df, strategy='mean'):
    handled_df = df.copy()
    numeric_cols = handled_df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        for col in numeric_cols:
            handled_df[col].fillna(handled_df[col].mean(), inplace=True)
    elif strategy == 'median':
        for col in numeric_cols:
            handled_df[col].fillna(handled_df[col].median(), inplace=True)
    elif strategy == 'mode':
        for col in numeric_cols:
            handled_df[col].fillna(handled_df[col].mode()[0], inplace=True)
    
    return handled_df

def clean_dataset(df, numeric_columns, outlier_removal=True, normalization='standardize', missing_strategy='mean'):
    df_cleaned = df.copy()
    
    if outlier_removal:
        df_cleaned = remove_outliers_iqr(df_cleaned, numeric_columns)
    
    if normalization == 'minmax':
        df_cleaned = normalize_minmax(df_cleaned, numeric_columns)
    elif normalization == 'standardize':
        df_cleaned = standardize_zscore(df_cleaned, numeric_columns)
    
    df_cleaned = handle_missing_values(df_cleaned, strategy=missing_strategy)
    
    return df_cleaned