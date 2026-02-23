
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    threshold (float): IQR multiplier (default 1.5)
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df.copy()

def normalize_minmax(dataframe, columns=None):
    """
    Normalize specified columns using min-max scaling.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    columns (list): List of column names to normalize (default: all numeric columns)
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    result_df = dataframe.copy()
    
    for col in columns:
        if col not in result_df.columns:
            continue
            
        if result_df[col].dtype in [np.float64, np.int64]:
            col_min = result_df[col].min()
            col_max = result_df[col].max()
            
            if col_max > col_min:
                result_df[col] = (result_df[col] - col_min) / (col_max - col_min)
    
    return result_df

def detect_skewed_columns(dataframe, skew_threshold=0.5):
    """
    Detect columns with significant skewness.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    skew_threshold (float): Absolute skewness threshold (default 0.5)
    
    Returns:
    dict: Dictionary with column names and their skewness values
    """
    skewed_cols = {}
    
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        skewness = stats.skew(dataframe[col].dropna())
        if abs(skewness) > skew_threshold:
            skewed_cols[col] = skewness
    
    return skewed_cols

def clean_dataset(dataframe, outlier_columns=None, normalize=True, skew_threshold=0.5):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    outlier_columns (list): Columns to remove outliers from (default: all numeric)
    normalize (bool): Whether to normalize columns (default: True)
    skew_threshold (float): Skewness detection threshold (default: 0.5)
    
    Returns:
    tuple: (cleaned DataFrame, cleaning report dictionary)
    """
    df_clean = dataframe.copy()
    report = {
        'original_shape': dataframe.shape,
        'outliers_removed': {},
        'skewed_columns': {},
        'normalized_columns': []
    }
    
    if outlier_columns is None:
        outlier_columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in outlier_columns:
        if col in df_clean.columns:
            original_len = len(df_clean)
            df_clean = remove_outliers_iqr(df_clean, col)
            removed = original_len - len(df_clean)
            report['outliers_removed'][col] = removed
    
    if normalize:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        df_clean = normalize_minmax(df_clean, numeric_cols)
        report['normalized_columns'] = numeric_cols
    
    skewed = detect_skewed_columns(df_clean, skew_threshold)
    report['skewed_columns'] = skewed
    report['final_shape'] = df_clean.shape
    
    return df_clean, report

def validate_dataframe(dataframe, required_columns=None, min_rows=10):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    dataframe (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(dataframe, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(dataframe) < min_rows:
        return False, f"DataFrame has less than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in dataframe.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    if dataframe.isnull().all().any():
        return False, "Some columns contain only null values"
    
    return True, "DataFrame validation passed"