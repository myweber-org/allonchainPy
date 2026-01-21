
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers from specified column using IQR method.
    Returns filtered dataframe and outlier indices.
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    mask = (dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)
    outliers = dataframe[~mask].index.tolist()
    
    return dataframe[mask].copy(), outliers

def normalize_minmax(dataframe, columns=None):
    """
    Normalize specified columns using min-max scaling.
    If columns is None, normalize all numeric columns.
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns
    
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col in dataframe.columns and np.issubdtype(dataframe[col].dtype, np.number):
            col_min = dataframe[col].min()
            col_max = dataframe[col].max()
            
            if col_max != col_min:
                normalized_df[col] = (dataframe[col] - col_min) / (col_max - col_min)
            else:
                normalized_df[col] = 0
    
    return normalized_df

def detect_skewed_columns(dataframe, skew_threshold=0.5):
    """
    Identify columns with significant skewness.
    Returns dictionary with column names and skewness values.
    """
    skewed_columns = {}
    
    for col in dataframe.select_dtypes(include=[np.number]).columns:
        skewness = stats.skew(dataframe[col].dropna())
        if abs(skewness) > skew_threshold:
            skewed_columns[col] = skewness
    
    return skewed_columns

def log_transform_skewed(dataframe, skewed_columns):
    """
    Apply log transformation to reduce skewness in specified columns.
    Handles zero and negative values by adding constant.
    """
    transformed_df = dataframe.copy()
    
    for col in skewed_columns:
        if col in dataframe.columns:
            col_data = dataframe[col].copy()
            min_val = col_data.min()
            
            if min_val <= 0:
                constant = abs(min_val) + 1
                col_data = col_data + constant
            
            transformed_df[col] = np.log1p(col_data)
    
    return transformed_df

def clean_dataset(dataframe, numeric_columns=None, outlier_threshold=1.5, normalize=True):
    """
    Comprehensive data cleaning pipeline.
    """
    cleaned_df = dataframe.copy()
    
    if numeric_columns is None:
        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
    
    all_outliers = []
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df, outliers = remove_outliers_iqr(cleaned_df, col, outlier_threshold)
            all_outliers.extend(outliers)
    
    skewed_cols = detect_skewed_columns(cleaned_df)
    if skewed_cols:
        cleaned_df = log_transform_skewed(cleaned_df, skewed_cols)
    
    if normalize:
        cleaned_df = normalize_minmax(cleaned_df, numeric_columns)
    
    return cleaned_df, all_outliers, skewed_cols

def validate_cleaned_data(dataframe, original_dataframe):
    """
    Validate that cleaning operations preserved data integrity.
    """
    validation_results = {
        'rows_removed': original_dataframe.shape[0] - dataframe.shape[0],
        'columns_preserved': set(dataframe.columns) == set(original_dataframe.columns),
        'no_nan_introduced': not dataframe.isnull().any().any(),
        'numeric_ranges_valid': True
    }
    
    for col in dataframe.select_dtypes(include=[np.number]).columns:
        if col in original_dataframe.columns:
            if dataframe[col].max() > 1.0 or dataframe[col].min() < 0.0:
                validation_results['numeric_ranges_valid'] = False
                break
    
    return validation_results