import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns=None, threshold=1.5):
    """
    Remove outliers using Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of columns to process, None for all numeric columns
    threshold (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_minmax(df, columns=None):
    """
    Normalize data using Min-Max scaling.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of columns to normalize, None for all numeric columns
    
    Returns:
    pd.DataFrame: Dataframe with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    for col in columns:
        min_val = df_normalized[col].min()
        max_val = df_normalized[col].max()
        
        if max_val > min_val:
            df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
    
    return df_normalized

def detect_skewed_columns(df, threshold=0.5):
    """
    Detect skewed columns using skewness coefficient.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    threshold (float): Absolute skewness threshold
    
    Returns:
    list: Columns with absolute skewness greater than threshold
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    skewed_cols = []
    
    for col in numeric_cols:
        skewness = stats.skew(df[col].dropna())
        if abs(skewness) > threshold:
            skewed_cols.append((col, skewness))
    
    return sorted(skewed_cols, key=lambda x: abs(x[1]), reverse=True)

def log_transform_skewed(df, skewed_cols):
    """
    Apply log transformation to reduce skewness.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    skewed_cols (list): List of column names to transform
    
    Returns:
    pd.DataFrame: Dataframe with transformed columns
    """
    df_transformed = df.copy()
    
    for col in skewed_cols:
        if col in df.columns:
            min_val = df[col].min()
            if min_val <= 0:
                df_transformed[col] = np.log1p(df[col] - min_val + 1)
            else:
                df_transformed[col] = np.log(df[col])
    
    return df_transformed

def clean_dataset(df, outlier_threshold=1.5, skew_threshold=0.5):
    """
    Complete data cleaning pipeline.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    outlier_threshold (float): IQR threshold for outlier removal
    skew_threshold (float): Skewness detection threshold
    
    Returns:
    pd.DataFrame: Cleaned and normalized dataframe
    """
    print(f"Original shape: {df.shape}")
    
    df_clean = remove_outliers_iqr(df, threshold=outlier_threshold)
    print(f"After outlier removal: {df_clean.shape}")
    
    skewed = detect_skewed_columns(df_clean, threshold=skew_threshold)
    skewed_cols = [col for col, _ in skewed]
    
    if skewed_cols:
        print(f"Skewed columns detected: {skewed_cols}")
        df_clean = log_transform_skewed(df_clean, skewed_cols)
    
    df_normalized = normalize_minmax(df_clean)
    print("Data normalization completed")
    
    return df_normalized
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        multiplier: IQR multiplier for outlier detection
        
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
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

def detect_skewness(data, column, threshold=0.5):
    """
    Detect skewness in data column.
    
    Args:
        data: pandas DataFrame
        column: column name to check
        threshold: absolute skewness threshold
        
    Returns:
        Tuple of (skewness_value, is_skewed)
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    skewness = data[column].skew()
    is_skewed = abs(skewness) > threshold
    
    return skewness, is_skewed

def apply_log_transform(data, column):
    """
    Apply log transformation to reduce skewness.
    
    Args:
        data: pandas DataFrame
        column: column name to transform
        
    Returns:
        Series with log-transformed values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if (data[column] <= 0).any():
        shifted_data = data[column] - data[column].min() + 1
        transformed = np.log(shifted_data)
    else:
        transformed = np.log(data[column])
    
    return transformed

def clean_dataset(data, numeric_columns=None, outlier_multiplier=1.5):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to clean
        outlier_multiplier: IQR multiplier for outlier detection
        
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column in cleaned_data.columns:
            # Remove outliers
            cleaned_data = remove_outliers_iqr(cleaned_data, column, outlier_multiplier)
            
            # Check and fix skewness
            skewness, is_skewed = detect_skewness(cleaned_data, column)
            
            if is_skewed:
                cleaned_data[f"{column}_log"] = apply_log_transform(cleaned_data, column)
            
            # Add normalized and standardized versions
            cleaned_data[f"{column}_normalized"] = normalize_minmax(cleaned_data, column)
            cleaned_data[f"{column}_standardized"] = standardize_zscore(cleaned_data, column)
    
    return cleaned_data