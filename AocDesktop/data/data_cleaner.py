import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): If True, remove duplicate rows.
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
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
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            elif fill_missing == 'median':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            mode_val = cleaned_df[col].mode()
            if not mode_val.empty:
                cleaned_df[col] = cleaned_df[col].fillna(mode_val.iloc[0])
    
    return cleaned_df
import pandas as pd
import re

def clean_string_column(series):
    """Standardize string column: lowercase, strip, remove extra spaces."""
    if series.dtype == 'object':
        series = series.astype(str)
        series = series.str.lower()
        series = series.str.strip()
        series = series.apply(lambda x: re.sub(r'\s+', ' ', x))
    return series

def remove_duplicates(df, subset=None, keep='first'):
    """Remove duplicate rows from DataFrame."""
    return df.drop_duplicates(subset=subset, keep=keep)

def clean_dataframe(df, string_columns=None):
    """Apply cleaning functions to DataFrame."""
    df_clean = df.copy()
    
    if string_columns:
        for col in string_columns:
            if col in df_clean.columns:
                df_clean[col] = clean_string_column(df_clean[col])
    else:
        for col in df_clean.select_dtypes(include=['object']).columns:
            df_clean[col] = clean_string_column(df_clean[col])
    
    df_clean = remove_duplicates(df_clean)
    return df_clean

def validate_email(email):
    """Basic email validation using regex."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email)))import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling
    """
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    return (data[column] - min_val) / (max_val - min_val)

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    return (data[column] - mean_val) / std_val

def clean_dataset(df, numeric_columns, outlier_threshold=1.5, normalization_method='standardize'):
    """
    Main cleaning function for numeric columns
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col not in cleaned_df.columns:
            continue
            
        # Remove outliers
        cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_threshold)
        
        # Apply normalization
        if normalization_method == 'minmax':
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
        elif normalization_method == 'standardize':
            cleaned_df[col] = standardize_zscore(cleaned_df, col)
    
    return cleaned_df.reset_index(drop=True)

def validate_data(df, required_columns):
    """
    Validate dataframe structure
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    return True

def generate_summary(df):
    """
    Generate cleaning summary statistics
    """
    summary = {
        'original_rows': len(df),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        'missing_values': df.isnull().sum().to_dict()
    }
    return summary