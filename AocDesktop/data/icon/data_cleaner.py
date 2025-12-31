import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=False, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (bool): Whether to fill missing values.
    fill_value: Value to use for filling missing data.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "Data validation passed"

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a specific column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to process.
    method (str): 'iqr' for interquartile range or 'zscore' for standard deviation.
    threshold (float): Threshold for outlier detection.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    data = df[column].copy()
    
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (data >= lower_bound) & (data <= upper_bound)
    elif method == 'zscore':
        mean = data.mean()
        std = data.std()
        z_scores = (data - mean) / std
        mask = abs(z_scores) <= threshold
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return df[mask]
import pandas as pd
import numpy as np

def remove_missing_rows(df, columns=None):
    """
    Remove rows with missing values from DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): Specific columns to check for missing values
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns:
        return df.dropna(subset=columns)
    return df.dropna()

def fill_missing_with_mean(df, columns=None):
    """
    Fill missing values with column mean.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): Specific columns to fill
    
    Returns:
        pd.DataFrame: DataFrame with filled values
    """
    df_copy = df.copy()
    
    if columns:
        for col in columns:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
    else:
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
    
    return df_copy

def remove_outliers_iqr(df, columns, factor=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): Columns to check for outliers
        factor (float): IQR multiplier
    
    Returns:
        pd.DataFrame: DataFrame without outliers
    """
    df_copy = df.copy()
    
    for col in columns:
        if col in df_copy.columns:
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            df_copy = df_copy[(df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)]
    
    return df_copy

def standardize_columns(df, columns):
    """
    Standardize specified columns to have zero mean and unit variance.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): Columns to standardize
    
    Returns:
        pd.DataFrame: DataFrame with standardized columns
    """
    df_copy = df.copy()
    
    for col in columns:
        if col in df_copy.columns:
            mean_val = df_copy[col].mean()
            std_val = df_copy[col].std()
            if std_val > 0:
                df_copy[col] = (df_copy[col] - mean_val) / std_val
    
    return df_copy

def clean_dataframe(df, missing_strategy='drop', outlier_columns=None):
    """
    Comprehensive data cleaning function.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        missing_strategy (str): 'drop' or 'fill'
        outlier_columns (list, optional): Columns to remove outliers from
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'drop':
        cleaned_df = remove_missing_rows(cleaned_df)
    elif missing_strategy == 'fill':
        cleaned_df = fill_missing_with_mean(cleaned_df)
    
    # Remove outliers if specified
    if outlier_columns:
        cleaned_df = remove_outliers_iqr(cleaned_df, outlier_columns)
    
    return cleaned_df