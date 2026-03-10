import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    if max_val == min_val:
        return df[column].apply(lambda x: 0.0)
    return df[column].apply(lambda x: (x - min_val) / (max_val - min_val))

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(df):
    required_checks = [
        (lambda d: isinstance(d, pd.DataFrame), "Input must be a pandas DataFrame"),
        (lambda d: not d.empty, "DataFrame cannot be empty"),
        (lambda d: d.isnull().sum().sum() == 0, "DataFrame contains null values")
    ]
    for check, message in required_checks:
        if not check(df):
            raise ValueError(message)
    return True
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None):
    """
    Clean a pandas DataFrame by removing duplicate rows and normalizing string columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    columns_to_clean (list, optional): List of column names to apply string normalization.
                                       If None, applies to all object dtype columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    
    # Determine columns to normalize
    if columns_to_clean is None:
        columns_to_clean = df_cleaned.select_dtypes(include=['object']).columns.tolist()
    
    # Normalize string columns
    for col in columns_to_clean:
        if col in df_cleaned.columns and df_cleaned[col].dtype == 'object':
            df_cleaned[col] = df_cleaned[col].apply(_normalize_string)
    
    return df_cleaned

def _normalize_string(value):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping special characters.
    
    Parameters:
    value: Input value (expected to be string).
    
    Returns:
    str: Normalized string, or original value if not a string.
    """
    if not isinstance(value, str):
        return value
    
    # Convert to lowercase
    normalized = value.lower()
    # Remove leading/trailing whitespace
    normalized = normalized.strip()
    # Replace multiple spaces with single space
    normalized = re.sub(r'\s+', ' ', normalized)
    # Remove non-alphanumeric characters except basic punctuation
    normalized = re.sub(r'[^\w\s.,!?-]', '', normalized)
    
    return normalized

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list, optional): List of required column names.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns=None, factor=1.5):
    """
    Remove outliers using Interquartile Range method.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to process (default: all numeric columns)
        factor: IQR multiplier for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_minmax(df, columns=None, feature_range=(0, 1)):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to normalize (default: all numeric columns)
        feature_range: tuple of (min, max) for output range
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    min_val, max_val = feature_range
    
    for col in columns:
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_max - col_min != 0:
                df_normalized[col] = (df[col] - col_min) / (col_max - col_min)
                df_normalized[col] = df_normalized[col] * (max_val - min_val) + min_val
    
    return df_normalized

def standardize_zscore(df, columns=None):
    """
    Standardize data using Z-score normalization.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to standardize (default: all numeric columns)
    
    Returns:
        DataFrame with standardized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_standardized = df.copy()
    
    for col in columns:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            
            if std_val != 0:
                df_standardized[col] = (df[col] - mean_val) / std_val
    
    return df_standardized

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        df: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'drop'
        columns: list of column names to process (default: all columns)
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = df.columns
    
    df_handled = df.copy()
    
    for col in columns:
        if col in df.columns and df[col].isnull().any():
            if strategy == 'drop':
                df_handled = df_handled.dropna(subset=[col])
            elif strategy == 'mean':
                df_handled[col] = df_handled[col].fillna(df[col].mean())
            elif strategy == 'median':
                df_handled[col] = df_handled[col].fillna(df[col].median())
            elif strategy == 'mode':
                df_handled[col] = df_handled[col].fillna(df[col].mode()[0])
    
    return df_handled.reset_index(drop=True)

def create_data_summary(df):
    """
    Create a comprehensive summary of the DataFrame.
    
    Args:
        df: pandas DataFrame
    
    Returns:
        Dictionary containing data summary statistics
    """
    summary = {
        'shape': df.shape,
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_stats': {},
        'categorical_stats': {}
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['numeric_stats'][col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'skewness': df[col].skew(),
            'kurtosis': df[col].kurtosis()
        }
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        summary['categorical_stats'][col] = {
            'unique_count': df[col].nunique(),
            'top_value': df[col].mode()[0] if not df[col].mode().empty else None,
            'top_count': df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0
        }
    
    return summary

def detect_anomalies(df, columns=None, method='zscore', threshold=3):
    """
    Detect anomalies in data.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to check (default: all numeric columns)
        method: 'zscore' or 'iqr'
        threshold: threshold for anomaly detection
    
    Returns:
        DataFrame with anomaly flags
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_anomaly = df.copy()
    
    for col in columns:
        if col in df.columns:
            if method == 'zscore':
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                anomaly_mask = z_scores > threshold
                df_anomaly[f'{col}_anomaly'] = False
                df_anomaly.loc[df[col].index[anomaly_mask], f'{col}_anomaly'] = True
            elif method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_anomaly[f'{col}_anomaly'] = ~df[col].between(lower_bound, upper_bound)
    
    return df_anomaly