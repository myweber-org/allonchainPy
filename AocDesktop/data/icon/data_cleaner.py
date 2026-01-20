import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    filtered_indices = np.where(z_scores < threshold)[0]
    filtered_data = data.iloc[filtered_indices].copy()
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data to [0, 1] range using min-max scaling.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        normalized = pd.Series([0.5] * len(data), index=data.index)
    else:
        normalized = (data[column] - min_val) / (max_val - min_val)
    
    result = data.copy()
    result[f'{column}_normalized'] = normalized
    return result

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        standardized = pd.Series([0] * len(data), index=data.index)
    else:
        standardized = (data[column] - mean_val) / std_val
    
    result = data.copy()
    result[f'{column}_standardized'] = standardized
    return result

def clean_dataset(data, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    """
    Apply outlier removal and normalization to multiple numeric columns.
    """
    cleaned_data = data.copy()
    
    for col in numeric_columns:
        if col not in cleaned_data.columns:
            print(f"Warning: Column '{col}' not found, skipping")
            continue
            
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, col)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, col)
        else:
            raise ValueError("outlier_method must be 'iqr' or 'zscore'")
    
    for col in numeric_columns:
        if col not in cleaned_data.columns:
            continue
            
        if normalize_method == 'minmax':
            cleaned_data = normalize_minmax(cleaned_data, col)
        elif normalize_method == 'zscore':
            cleaned_data = normalize_zscore(cleaned_data, col)
        else:
            raise ValueError("normalize_method must be 'minmax' or 'zscore'")
    
    return cleaned_data

def get_summary_statistics(data, numeric_columns):
    """
    Calculate summary statistics for numeric columns.
    """
    summary = {}
    
    for col in numeric_columns:
        if col not in data.columns:
            continue
            
        col_data = data[col].dropna()
        summary[col] = {
            'count': len(col_data),
            'mean': col_data.mean(),
            'std': col_data.std(),
            'min': col_data.min(),
            '25%': col_data.quantile(0.25),
            'median': col_data.median(),
            '75%': col_data.quantile(0.75),
            'max': col_data.max(),
            'missing': data[col].isnull().sum()
        }
    
    return pd.DataFrame(summary).T

def example_usage():
    """
    Example demonstrating how to use the data cleaning functions.
    """
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 200),
        'feature_b': np.random.exponential(50, 200),
        'category': np.random.choice(['A', 'B', 'C'], 200)
    })
    
    print("Original data shape:", sample_data.shape)
    print("\nOriginal summary statistics:")
    print(get_summary_statistics(sample_data, ['feature_a', 'feature_b']))
    
    cleaned = clean_dataset(
        sample_data, 
        numeric_columns=['feature_a', 'feature_b'],
        outlier_method='iqr',
        normalize_method='minmax'
    )
    
    print("\nCleaned data shape:", cleaned.shape)
    print("\nCleaned summary statistics:")
    print(get_summary_statistics(cleaned, ['feature_a', 'feature_b', 'feature_a_normalized', 'feature_b_normalized']))
    
    return cleaned

if __name__ == "__main__":
    result = example_usage()
    print("\nExample completed successfully.")
    print(f"Final dataset has {result.shape[0]} rows and {result.shape[1]} columns.")