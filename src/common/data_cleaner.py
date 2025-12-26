
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.

    Parameters:
    df (pd.DataFrame): The input DataFrame to clean.
    drop_duplicates (bool): If True, remove duplicate rows.
    fill_missing (str): Method to fill missing values.
                        Options: 'mean', 'median', 'mode', or 'drop'.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    cleaned_df = df.copy()

    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()

    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            elif fill_missing == 'median':
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None, inplace=True)

    return cleaned_dfimport numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Args:
        data: pandas DataFrame
        column: Column name to process
        multiplier: IQR multiplier for outlier detection
    
    Returns:
        DataFrame with outliers removed
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

def z_score_normalize(data, column):
    """
    Normalize a column using z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: Column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    normalized = (data[column] - mean_val) / std_val
    return normalized

def min_max_normalize(data, column, feature_range=(0, 1)):
    """
    Normalize a column using min-max scaling.
    
    Args:
        data: pandas DataFrame
        column: Column name to normalize
        feature_range: Desired range of transformed data
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([feature_range[0]] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    normalized = normalized * (feature_range[1] - feature_range[0]) + feature_range[0]
    return normalized

def detect_skewed_columns(data, threshold=0.5):
    """
    Identify columns with skewed distributions.
    
    Args:
        data: pandas DataFrame
        threshold: Absolute skewness threshold for detection
    
    Returns:
        Dictionary with column names and their skewness values
    """
    skewed_cols = {}
    
    for col in data.select_dtypes(include=[np.number]).columns:
        skewness = stats.skew(data[col].dropna())
        if abs(skewness) > threshold:
            skewed_cols[col] = skewness
    
    return skewed_cols

def log_transform(data, column):
    """
    Apply log transformation to reduce skewness.
    
    Args:
        data: pandas DataFrame
        column: Column name to transform
    
    Returns:
        Series with log-transformed values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if (data[column] <= 0).any():
        transformed = np.log1p(data[column] - data[column].min() + 1)
    else:
        transformed = np.log(data[column])
    
    return transformed

def clean_dataset(data, numeric_columns=None, outlier_multiplier=1.5, normalize_method='zscore'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: Input DataFrame
        numeric_columns: List of numeric columns to process
        outlier_multiplier: Multiplier for IQR outlier detection
        normalize_method: 'zscore' or 'minmax' normalization
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for col in numeric_columns:
        if col in cleaned_data.columns:
            # Remove outliers
            cleaned_data = remove_outliers_iqr(cleaned_data, col, outlier_multiplier)
            
            # Normalize
            if normalize_method == 'zscore':
                cleaned_data[f'{col}_normalized'] = z_score_normalize(cleaned_data, col)
            elif normalize_method == 'minmax':
                cleaned_data[f'{col}_normalized'] = min_max_normalize(cleaned_data, col)
    
    return cleaned_dataimport pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def fill_missing_with_mean(df, column):
    mean_value = df[column].mean()
    df[column] = df[column].fillna(mean_value)
    return df

def standardize_column(df, column):
    mean = df[column].mean()
    std = df[column].std()
    df[column] = (df[column] - mean) / std
    return df

def clean_dataset(df, numeric_columns):
    for col in numeric_columns:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)
            df = fill_missing_with_mean(df, col)
            df = standardize_column(df, col)
    return df.reset_index(drop=True)
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
        'count': df[column].count()
    }
    
    return stats

def clean_dataset(df, numeric_columns=None):
    """
    Clean a dataset by removing outliers from all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Create sample data with outliers
    data = {
        'temperature': np.concatenate([
            np.random.normal(25, 5, 90),
            np.array([-10, 60, 100])
        ]),
        'humidity': np.concatenate([
            np.random.normal(50, 15, 90),
            np.array([-5, 120, 150])
        ]),
        'pressure': np.random.normal(1013, 10, 93)
    }
    
    sample_df = pd.DataFrame(data)
    
    print("Original dataset shape:", sample_df.shape)
    print("\nOriginal summary statistics:")
    for col in sample_df.columns:
        stats = calculate_summary_statistics(sample_df, col)
        print(f"\n{col}:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}")
    
    # Clean the dataset
    cleaned_df = clean_dataset(sample_df)
    
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("\nCleaned summary statistics:")
    for col in cleaned_df.columns:
        stats = calculate_summary_statistics(cleaned_df, col)
        print(f"\n{col}:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}")