
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None):
    """
    Clean a pandas DataFrame by removing duplicate rows and normalizing string columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_clean (list, optional): List of column names to apply string normalization.
                                          If None, all object dtype columns are cleaned.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with duplicates removed and strings normalized.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    
    # Determine which columns to normalize
    if columns_to_clean is None:
        columns_to_clean = df_cleaned.select_dtypes(include=['object']).columns.tolist()
    
    # Apply string normalization to specified columns
    for col in columns_to_clean:
        if col in df_cleaned.columns and df_cleaned[col].dtype == 'object':
            df_cleaned[col] = df_cleaned[col].apply(_normalize_string)
    
    return df_cleaned

def _normalize_string(value):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping special characters from the beginning and end.
    
    Args:
        value: Input value to normalize. If not a string, returns as-is.
    
    Returns:
        Normalized string or original value.
    """
    if not isinstance(value, str):
        return value
    
    # Convert to lowercase
    normalized = value.lower()
    
    # Remove leading/trailing whitespace
    normalized = normalized.strip()
    
    # Replace multiple spaces with single space
    normalized = re.sub(r'\s+', ' ', normalized)
    
    return normalized

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column using a simple regex pattern.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        email_column (str): Name of the column containing email addresses.
    
    Returns:
        pd.DataFrame: DataFrame with an additional 'email_valid' boolean column.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    df = df.copy()
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['email_valid'] = df[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notnull(x) else False
    )
    
    return df
import pandas as pd
import numpy as np

def remove_missing_rows(df, columns=None):
    """
    Remove rows with missing values from DataFrame.
    If columns specified, only check those columns.
    """
    if columns:
        return df.dropna(subset=columns)
    return df.dropna()

def fill_missing_with_mean(df, columns):
    """
    Fill missing values in specified columns with column mean.
    """
    df_filled = df.copy()
    for col in columns:
        if col in df.columns:
            mean_val = df[col].mean()
            df_filled[col] = df_filled[col].fillna(mean_val)
    return df_filled

def detect_outliers_iqr(df, column, multiplier=1.5):
    """
    Detect outliers using IQR method.
    Returns boolean Series indicating outliers.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def cap_outliers(df, column, method='iqr', multiplier=1.5):
    """
    Cap outliers to specified bounds.
    Supports IQR method for outlier detection.
    """
    df_capped = df.copy()
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        df_capped[column] = np.where(df_capped[column] < lower_bound, lower_bound, df_capped[column])
        df_capped[column] = np.where(df_capped[column] > upper_bound, upper_bound, df_capped[column])
    
    return df_capped

def standardize_column(df, column):
    """
    Standardize column to have mean=0 and std=1.
    """
    df_standardized = df.copy()
    mean_val = df[column].mean()
    std_val = df[column].std()
    
    if std_val > 0:
        df_standardized[column] = (df[column] - mean_val) / std_val
    
    return df_standardized

def normalize_column(df, column):
    """
    Normalize column to range [0, 1].
    """
    df_normalized = df.copy()
    min_val = df[column].min()
    max_val = df[column].max()
    
    if max_val > min_val:
        df_normalized[column] = (df[column] - min_val) / (max_val - min_val)
    
    return df_normalized

def clean_dataset(df, missing_strategy='remove', outlier_strategy='cap'):
    """
    Comprehensive data cleaning pipeline.
    """
    df_clean = df.copy()
    
    # Handle missing values
    if missing_strategy == 'remove':
        df_clean = remove_missing_rows(df_clean)
    elif missing_strategy == 'mean':
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean = fill_missing_with_mean(df_clean, numeric_cols)
    
    # Handle outliers for numeric columns
    if outlier_strategy == 'cap':
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_clean = cap_outliers(df_clean, col)
    
    return df_clean