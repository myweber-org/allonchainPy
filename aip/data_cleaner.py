
def remove_duplicates(data_list):
    """
    Remove duplicate entries from a list while preserving order.
    
    Args:
        data_list: A list of elements (must be hashable)
    
    Returns:
        A new list with duplicates removed
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
        data_list: A list of elements
        key_func: Function to extract comparison key (optional)
    
    Returns:
        A new list with duplicates removed based on key
    """
    if key_func is None:
        return remove_duplicates(data_list)
    
    seen = set()
    result = []
    
    for item in data_list:
        key = key_func(item)
        if key not in seen:
            seen.add(key)
            result.append(item)
    
    return result

if __name__ == "__main__":
    # Example usage
    sample_data = [1, 2, 2, 3, 4, 4, 5]
    cleaned = remove_duplicates(sample_data)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {cleaned}")
    
    # Example with custom key
    data_with_dicts = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
        {"id": 1, "name": "Alice"},
        {"id": 3, "name": "Charlie"}
    ]
    
    cleaned_dicts = clean_data_with_key(data_with_dicts, key_func=lambda x: x["id"])
    print(f"\nOriginal dicts: {data_with_dicts}")
    print(f"Cleaned dicts: {cleaned_dicts}")import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list, optional): Column labels to consider for duplicates.
        keep (str, optional): Which duplicates to keep.
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows.")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): Columns that must be present.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame.")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty.")
        return True
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    return True

def clean_numeric_column(df, column_name, fill_method='mean'):
    """
    Clean numeric column by handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Name of column to clean.
        fill_method (str): Method to fill missing values ('mean', 'median', 'zero').
    
    Returns:
        pd.DataFrame: DataFrame with cleaned column.
    """
    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' not found in DataFrame.")
        return df
    
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        print(f"Error: Column '{column_name}' is not numeric.")
        return df
    
    missing_count = df[column_name].isna().sum()
    
    if missing_count > 0:
        if fill_method == 'mean':
            fill_value = df[column_name].mean()
        elif fill_method == 'median':
            fill_value = df[column_name].median()
        elif fill_method == 'zero':
            fill_value = 0
        else:
            print(f"Warning: Unknown fill method '{fill_method}', using mean.")
            fill_value = df[column_name].mean()
        
        df[column_name] = df[column_name].fillna(fill_value)
        print(f"Filled {missing_count} missing values in '{column_name}' with {fill_method}.")
    
    return df

def process_dataframe(df, operations=None):
    """
    Apply multiple cleaning operations to a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        operations (list, optional): List of cleaning operations to apply.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    if not validate_dataframe(df):
        return df
    
    result_df = df.copy()
    
    if operations:
        for operation in operations:
            if operation['type'] == 'remove_duplicates':
                result_df = remove_duplicates(
                    result_df, 
                    subset=operation.get('subset'), 
                    keep=operation.get('keep', 'first')
                )
            elif operation['type'] == 'clean_numeric':
                result_df = clean_numeric_column(
                    result_df,
                    column_name=operation['column'],
                    fill_method=operation.get('fill_method', 'mean')
                )
    
    return result_df
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

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns
    """
    if columns is None:
        columns = data.columns
    
    data_copy = data.copy()
    
    for col in columns:
        if col not in data_copy.columns:
            continue
            
        missing_count = data_copy[col].isnull().sum()
        if missing_count == 0:
            continue
            
        if strategy == 'mean':
            fill_value = data_copy[col].mean()
        elif strategy == 'median':
            fill_value = data_copy[col].median()
        elif strategy == 'mode':
            fill_value = data_copy[col].mode()[0]
        elif strategy == 'drop':
            data_copy = data_copy.dropna(subset=[col])
            continue
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        data_copy[col] = data_copy[col].fillna(fill_value)
    
    return data_copy

def detect_skewed_columns(data, threshold=0.5):
    """
    Detect columns with skewed distributions
    """
    skewed_columns = []
    
    for col in data.select_dtypes(include=[np.number]).columns:
        skewness = stats.skew(data[col].dropna())
        if abs(skewness) > threshold:
            skewed_columns.append((col, skewness))
    
    return sorted(skewed_columns, key=lambda x: abs(x[1]), reverse=True)