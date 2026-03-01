
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    cleaned_df = cleaned_df.dropna()
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

def validate_data(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return True
def remove_duplicates(data_list):
    """
    Remove duplicate entries from a list while preserving order.
    
    Args:
        data_list: A list of elements (must be hashable).
    
    Returns:
        A new list with duplicates removed.
    """
    seen = set()
    result = []
    for item in data_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_data_with_key(data_list, key_func=None):
    """
    Remove duplicates based on a key function.
    
    Args:
        data_list: A list of elements.
        key_func: A function that returns a key for each element.
                  If None, uses the element itself.
    
    Returns:
        A new list with duplicates removed based on the key.
    """
    seen = set()
    result = []
    for item in data_list:
        key = key_func(item) if key_func else item
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, 4, 4, 5]
    cleaned = remove_duplicates(sample_data)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {cleaned}")
    
    sample_dicts = [{"id": 1}, {"id": 2}, {"id": 1}, {"id": 3}]
    cleaned_dicts = clean_data_with_key(sample_dicts, key_func=lambda x: x["id"])
    print(f"Original dicts: {sample_dicts}")
    print(f"Cleaned dicts: {cleaned_dicts}")import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas Series using the IQR method.
    Returns cleaned Series and outlier indices.
    """
    if not isinstance(data, pd.Series):
        raise TypeError("Input data must be a pandas Series")
    
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_mask = (data < lower_bound) | (data > upper_bound)
    cleaned_data = data[~outlier_mask].copy()
    
    return cleaned_data, data[outlier_mask].index.tolist()

def normalize_minmax(data):
    """
    Normalize data to range [0, 1] using min-max scaling.
    Handles NaN values by ignoring them in calculation.
    """
    if not isinstance(data, (pd.Series, np.ndarray, list)):
        raise TypeError("Data must be Series, array, or list")
    
    data_array = np.array(data, dtype=float)
    valid_mask = ~np.isnan(data_array)
    
    if not np.any(valid_mask):
        return np.full_like(data_array, np.nan)
    
    valid_data = data_array[valid_mask]
    data_min = np.min(valid_data)
    data_max = np.max(valid_data)
    
    if data_max == data_min:
        normalized = np.zeros_like(data_array)
    else:
        normalized = (data_array - data_min) / (data_max - data_min)
    
    normalized[~valid_mask] = np.nan
    return normalized

def clean_dataframe(df, numeric_columns=None):
    """
    Clean a DataFrame by removing outliers and normalizing numeric columns.
    Returns cleaned DataFrame and outlier statistics.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    outlier_report = {}
    
    for col in numeric_columns:
        if col not in df.columns:
            continue
            
        original_series = df[col]
        cleaned_series, outlier_indices = remove_outliers_iqr(original_series, col)
        
        if outlier_indices:
            cleaned_df.loc[outlier_indices, col] = np.nan
            outlier_report[col] = {
                'count': len(outlier_indices),
                'indices': outlier_indices,
                'percentage': len(outlier_indices) / len(df) * 100
            }
        
        normalized_values = normalize_minmax(cleaned_df[col])
        cleaned_df[col] = normalized_values
    
    return cleaned_df, outlier_report

def validate_data_quality(df, threshold=0.1):
    """
    Validate data quality by checking for excessive missing values.
    Returns boolean indicating if data passes quality check.
    """
    missing_ratio = df.isnull().sum() / len(df)
    high_missing = missing_ratio[missing_ratio > threshold].index.tolist()
    
    if high_missing:
        print(f"Warning: Columns with >{threshold*100}% missing values: {high_missing}")
        return False
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 1, 1000)
    })
    
    sample_data.loc[::100, 'A'] = np.nan
    sample_data.loc[50:55, 'B'] = 1000
    
    cleaned, report = clean_dataframe(sample_data)
    quality_ok = validate_data_quality(cleaned)
    
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned.shape}")
    print(f"Outliers found: {len(report)} columns")
    print(f"Data quality check passed: {quality_ok}")