import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers from a pandas Series using the IQR method.
    
    Parameters:
    data (pd.Series): Input data series
    column (str): Column name to process
    multiplier (float): IQR multiplier for outlier detection
    
    Returns:
    pd.Series: Data with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    series = data[column]
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    return data[(series >= lower_bound) & (series <= upper_bound)]

def normalize_minmax(data, column):
    """
    Normalize data to [0, 1] range using min-max scaling.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to normalize
    
    Returns:
    pd.Series: Normalized data series
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    series = data[column]
    min_val = series.min()
    max_val = series.max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(series), index=series.index)
    
    return (series - min_val) / (max_val - min_val)

def z_score_normalize(data, column):
    """
    Normalize data using z-score standardization.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to normalize
    
    Returns:
    pd.Series: Z-score normalized data series
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    series = data[column]
    mean = series.mean()
    std = series.std()
    
    if std == 0:
        return pd.Series([0] * len(series), index=series.index)
    
    return (series - mean) / std

def clean_dataset(df, numeric_columns=None, outlier_multiplier=1.5, normalization_method='minmax'):
    """
    Comprehensive data cleaning pipeline for numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    numeric_columns (list): List of numeric column names to process
    outlier_multiplier (float): IQR multiplier for outlier removal
    normalization_method (str): 'minmax' or 'zscore' normalization method
    
    Returns:
    pd.DataFrame: Cleaned dataframe with processed numeric columns
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col not in df.columns:
            continue
            
        # Remove outliers
        cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_multiplier)
        
        # Apply normalization
        if normalization_method == 'minmax':
            cleaned_df[f'{col}_normalized'] = normalize_minmax(cleaned_df, col)
        elif normalization_method == 'zscore':
            cleaned_df[f'{col}_normalized'] = z_score_normalize(cleaned_df, col)
        else:
            raise ValueError("Normalization method must be 'minmax' or 'zscore'")
    
    return cleaned_df

def calculate_statistics(df, column):
    """
    Calculate descriptive statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    series = df[column]
    
    stats_dict = {
        'mean': series.mean(),
        'median': series.median(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max(),
        'q1': series.quantile(0.25),
        'q3': series.quantile(0.75),
        'skewness': series.skew(),
        'kurtosis': series.kurtosis()
    }
    
    return stats_dict

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    # Add some outliers
    sample_data.loc[10, 'feature_a'] = 500
    sample_data.loc[20, 'feature_b'] = 1000
    
    print("Original data shape:", sample_data.shape)
    print("\nOriginal statistics for feature_a:")
    print(calculate_statistics(sample_data, 'feature_a'))
    
    # Clean the dataset
    cleaned_data = clean_dataset(
        sample_data, 
        numeric_columns=['feature_a', 'feature_b'],
        outlier_multiplier=1.5,
        normalization_method='minmax'
    )
    
    print("\nCleaned data shape:", cleaned_data.shape)
    print("\nCleaned statistics for feature_a:")
    print(calculate_statistics(cleaned_data, 'feature_a'))
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    outliers_removed = len(data) - len(filtered_data)
    
    return filtered_data, outliers_removed

def z_score_normalize(data, column):
    """
    Normalize data using z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column]
    
    normalized = (data[column] - mean_val) / std_val
    return normalized

def min_max_normalize(data, column, feature_range=(0, 1)):
    """
    Normalize data using min-max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column]
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    
    if feature_range != (0, 1):
        min_target, max_target = feature_range
        normalized = normalized * (max_target - min_target) + min_target
    
    return normalized

def clean_dataset(data, numeric_columns=None, outlier_factor=1.5, normalize_method='zscore'):
    """
    Comprehensive data cleaning function
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    stats_report = {}
    
    for col in numeric_columns:
        if col not in data.columns:
            continue
            
        original_len = len(cleaned_data)
        cleaned_data, outliers = remove_outliers_iqr(cleaned_data, col, outlier_factor)
        stats_report[col] = {'outliers_removed': outliers}
        
        if normalize_method == 'zscore':
            cleaned_data[f'{col}_normalized'] = z_score_normalize(cleaned_data, col)
        elif normalize_method == 'minmax':
            cleaned_data[f'{col}_normalized'] = min_max_normalize(cleaned_data, col)
        else:
            raise ValueError("normalize_method must be 'zscore' or 'minmax'")
    
    return cleaned_data, stats_report

def validate_data(data, required_columns=None, allow_nan=False):
    """
    Validate dataset structure and content
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if not allow_nan:
        nan_count = data.isnull().sum().sum()
        if nan_count > 0:
            raise ValueError(f"Dataset contains {nan_count} NaN values")
    
    return True