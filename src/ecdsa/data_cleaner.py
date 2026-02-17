
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        factor: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling to [0, 1] range.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values in numeric columns.
    
    Args:
        data: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        DataFrame with handled missing values
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        return data.dropna(subset=numeric_cols).copy()
    
    result = data.copy()
    
    for col in numeric_cols:
        if result[col].isnull().any():
            if strategy == 'mean':
                fill_value = result[col].mean()
            elif strategy == 'median':
                fill_value = result[col].median()
            elif strategy == 'mode':
                fill_value = result[col].mode()[0]
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            result[col] = result[col].fillna(fill_value)
    
    return result

def create_sample_data():
    """
    Create sample data for testing the cleaning functions.
    
    Returns:
        DataFrame with sample data containing outliers and missing values
    """
    np.random.seed(42)
    n_samples = 100
    
    data = pd.DataFrame({
        'temperature': np.random.normal(25, 5, n_samples),
        'humidity': np.random.uniform(30, 80, n_samples),
        'pressure': np.random.normal(1013, 10, n_samples)
    })
    
    data.loc[10:15, 'temperature'] = np.nan
    data.loc[95, 'temperature'] = 150
    data.loc[96, 'humidity'] = -10
    
    return data

if __name__ == "__main__":
    sample_data = create_sample_data()
    print("Original data shape:", sample_data.shape)
    print("\nMissing values per column:")
    print(sample_data.isnull().sum())
    
    cleaned_data = handle_missing_values(sample_data, strategy='mean')
    cleaned_data = remove_outliers_iqr(cleaned_data, 'temperature')
    cleaned_data = remove_outliers_iqr(cleaned_data, 'humidity')
    
    print("\nCleaned data shape:", cleaned_data.shape)
    print("\nTemperature statistics after cleaning:")
    print(f"Mean: {cleaned_data['temperature'].mean():.2f}")
    print(f"Std: {cleaned_data['temperature'].std():.2f}")
    
    normalized_temp = normalize_minmax(cleaned_data, 'temperature')
    standardized_temp = standardize_zscore(cleaned_data, 'temperature')
    
    print(f"\nNormalized temperature range: [{normalized_temp.min():.3f}, {normalized_temp.max():.3f}]")
    print(f"Standardized temperature mean: {standardized_temp.mean():.3f}, std: {standardized_temp.std():.3f}")