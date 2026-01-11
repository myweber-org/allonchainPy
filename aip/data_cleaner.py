
import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method.
    Returns indices of outliers.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)].index
    return outliers.tolist()

def remove_outliers(data, column, threshold=1.5):
    """
    Remove outliers from specified column using IQR method.
    Returns cleaned DataFrame.
    """
    outliers = detect_outliers_iqr(data, column, threshold)
    cleaned_data = data.drop(outliers)
    return cleaned_data

def normalize_minmax(data, column):
    """
    Normalize column values to range [0, 1] using min-max scaling.
    Returns Series with normalized values.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize column values using z-score normalization.
    Returns Series with standardized values.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    Supported strategies: 'mean', 'median', 'mode', 'drop'
    """
    if columns is None:
        columns = data.columns
    
    data_copy = data.copy()
    
    for col in columns:
        if col not in data_copy.columns:
            continue
            
        if data_copy[col].isnull().sum() == 0:
            continue
        
        if strategy == 'drop':
            data_copy = data_copy.dropna(subset=[col])
        elif strategy == 'mean':
            fill_value = data_copy[col].mean()
            data_copy[col] = data_copy[col].fillna(fill_value)
        elif strategy == 'median':
            fill_value = data_copy[col].median()
            data_copy[col] = data_copy[col].fillna(fill_value)
        elif strategy == 'mode':
            fill_value = data_copy[col].mode()[0]
            data_copy[col] = data_copy[col].fillna(fill_value)
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")
    
    return data_copy

def validate_data_types(data, schema):
    """
    Validate data types according to provided schema.
    Schema format: {'column_name': 'expected_type'}
    Returns list of validation errors.
    """
    errors = []
    
    for column, expected_type in schema.items():
        if column not in data.columns:
            errors.append(f"Column '{column}' not found in data")
            continue
        
        actual_type = str(data[column].dtype)
        
        type_mapping = {
            'int': ['int64', 'int32', 'int16', 'int8'],
            'float': ['float64', 'float32'],
            'str': ['object'],
            'bool': ['bool'],
            'datetime': ['datetime64[ns]']
        }
        
        if expected_type in type_mapping:
            if actual_type not in type_mapping[expected_type]:
                errors.append(f"Column '{column}' expected {expected_type}, got {actual_type}")
        elif actual_type != expected_type:
            errors.append(f"Column '{column}' expected {expected_type}, got {actual_type}")
    
    return errors

def create_summary_statistics(data):
    """
    Create comprehensive summary statistics for numerical columns.
    Returns DataFrame with statistics.
    """
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) == 0:
        return pd.DataFrame()
    
    summary = pd.DataFrame({
        'mean': data[numerical_cols].mean(),
        'median': data[numerical_cols].median(),
        'std': data[numerical_cols].std(),
        'min': data[numerical_cols].min(),
        'max': data[numerical_cols].max(),
        'q1': data[numerical_cols].quantile(0.25),
        'q3': data[numerical_cols].quantile(0.75),
        'skewness': data[numerical_cols].apply(lambda x: stats.skew(x.dropna())),
        'kurtosis': data[numerical_cols].apply(lambda x: stats.kurtosis(x.dropna())),
        'missing': data[numerical_cols].isnull().sum(),
        'zeros': (data[numerical_cols] == 0).sum()
    })
    
    return summary