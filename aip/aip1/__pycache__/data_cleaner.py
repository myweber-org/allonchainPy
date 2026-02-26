import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers from a pandas Series using the IQR method.
    """
    if not isinstance(data, pd.Series):
        series = pd.Series(data)
    else:
        series = data.copy()
    
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_series = series[(series >= lower_bound) & (series <= upper_bound)]
    return filtered_series

def z_score_normalize(data):
    """
    Normalize data using z-score normalization.
    """
    if isinstance(data, pd.Series):
        array_data = data.values
    else:
        array_data = np.array(data)
    
    mean_val = np.mean(array_data)
    std_val = np.std(array_data)
    
    if std_val == 0:
        return np.zeros_like(array_data)
    
    normalized = (array_data - mean_val) / std_val
    return normalized

def min_max_normalize(data, feature_range=(0, 1)):
    """
    Normalize data to a specified range using min-max scaling.
    """
    if isinstance(data, pd.Series):
        array_data = data.values
    else:
        array_data = np.array(data)
    
    min_val = np.min(array_data)
    max_val = np.max(array_data)
    
    if max_val == min_val:
        return np.full_like(array_data, feature_range[0])
    
    normalized = (array_data - min_val) / (max_val - min_val)
    normalized = normalized * (feature_range[1] - feature_range[0]) + feature_range[0]
    return normalized

def clean_dataframe(df, numeric_columns=None, outlier_method='iqr', normalize_method='zscore'):
    """
    Clean a DataFrame by handling outliers and normalizing numeric columns.
    """
    df_clean = df.copy()
    
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_columns:
        if col not in df.columns:
            continue
            
        if outlier_method == 'iqr':
            df_clean[col] = remove_outliers_iqr(df_clean[col], col)
        elif outlier_method == 'zscore':
            z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
            df_clean = df_clean[(z_scores < 3)]
        
        if normalize_method == 'zscore':
            df_clean[col] = z_score_normalize(df_clean[col])
        elif normalize_method == 'minmax':
            df_clean[col] = min_max_normalize(df_clean[col])
    
    return df_clean.dropna()

def validate_data(data, check_finite=True, check_non_negative=False):
    """
    Validate data for common issues.
    """
    if isinstance(data, pd.DataFrame):
        array_data = data.values
    elif isinstance(data, pd.Series):
        array_data = data.values
    else:
        array_data = np.array(data)
    
    if check_finite:
        if not np.all(np.isfinite(array_data)):
            raise ValueError("Data contains non-finite values (inf or nan)")
    
    if check_non_negative:
        if np.any(array_data < 0):
            raise ValueError("Data contains negative values")
    
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(2, 1000),
        'C': np.random.uniform(0, 1, 1000)
    })
    
    print("Original data shape:", sample_data.shape)
    print("Original data stats:")
    print(sample_data.describe())
    
    cleaned_data = clean_dataframe(sample_data, normalize_method='minmax')
    print("\nCleaned data shape:", cleaned_data.shape)
    print("Cleaned data stats:")
    print(cleaned_data.describe())
    
    try:
        validate_data(cleaned_data, check_finite=True, check_non_negative=True)
        print("\nData validation passed")
    except ValueError as e:
        print(f"\nData validation failed: {e}")