
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using Interquartile Range method
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

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    filtered_indices = np.where(z_scores < threshold)[0]
    filtered_data = data.iloc[filtered_indices].copy()
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].copy()
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].copy()
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns
    """
    if columns is None:
        columns = data.columns
    
    data_copy = data.copy()
    
    for column in columns:
        if column not in data_copy.columns:
            continue
            
        if data_copy[column].isnull().any():
            if strategy == 'mean':
                fill_value = data_copy[column].mean()
            elif strategy == 'median':
                fill_value = data_copy[column].median()
            elif strategy == 'mode':
                fill_value = data_copy[column].mode()[0]
            elif strategy == 'drop':
                data_copy = data_copy.dropna(subset=[column])
                continue
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            data_copy[column] = data_copy[column].fillna(fill_value)
    
    return data_copy

def validate_data_types(data, expected_types):
    """
    Validate that columns have expected data types
    """
    validation_results = {}
    
    for column, expected_type in expected_types.items():
        if column not in data.columns:
            validation_results[column] = {'valid': False, 'message': 'Column not found'}
            continue
        
        actual_type = data[column].dtype
        if actual_type != expected_type:
            try:
                data[column].astype(expected_type)
                validation_results[column] = {'valid': True, 'message': f'Convertible from {actual_type} to {expected_type}'}
            except:
                validation_results[column] = {'valid': False, 'message': f'Cannot convert from {actual_type} to {expected_type}'}
        else:
            validation_results[column] = {'valid': True, 'message': f'Already {expected_type}'}
    
    return validation_results
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    cleaned_df = df.copy()

    if drop_duplicates:
        initial_rows = cleaned_df.shape[0]
        cleaned_df.drop_duplicates(inplace=True)
        removed = initial_rows - cleaned_df.shape[0]
        print(f"Removed {removed} duplicate rows.")

    if cleaned_df.isnull().sum().any():
        print("Handling missing values...")
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = cleaned_df.select_dtypes(exclude=[np.number]).columns

        if fill_missing == 'mean' and len(numeric_cols) > 0:
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
        elif fill_missing == 'median' and len(numeric_cols) > 0:
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
        elif fill_missing == 'zero' and len(numeric_cols) > 0:
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(0)

        for col in non_numeric_cols:
            if cleaned_df[col].isnull().any():
                cleaned_df[col].fillna('Unknown', inplace=True)

    print(f"Data cleaning complete. Final shape: {cleaned_df.shape}")
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, np.nan, 5],
        'B': [10.5, np.nan, 10.5, 13.2, 15.0],
        'C': ['X', 'Y', 'X', np.nan, 'Z']
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    cleaned = clean_dataframe(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to process
    
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
    
    return filtered_df.reset_index(drop=True)

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a DataFrame column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to analyze
    
    Returns:
        dict: Dictionary containing statistical measures
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
    Clean numeric data by removing outliers from specified columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): List of column names to clean. 
                                 If None, cleans all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for column in columns:
        if column in cleaned_df.columns:
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
            except Exception as e:
                print(f"Warning: Could not process column '{column}': {e}")
    
    return cleaned_df