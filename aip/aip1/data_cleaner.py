
import pandas as pd

def clean_dataset(df, remove_duplicates=True, fill_na=None):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        remove_duplicates (bool): Whether to remove duplicate rows
        fill_na (str or scalar): Method to fill NaN values ('mean', 'median', 'mode', or scalar value)
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Handle missing values
    if fill_na is not None:
        if fill_na == 'mean':
            df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))
        elif fill_na == 'median':
            df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
        elif fill_na == 'mode':
            df_clean = df_clean.fillna(df_clean.mode().iloc[0])
        else:
            df_clean = df_clean.fillna(fill_na)
    
    # Remove duplicates
    if remove_duplicates:
        df_clean = df_clean.drop_duplicates()
    
    return df_clean

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of column names that must be present
        min_rows (int): Minimum number of rows required
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"

def normalize_columns(df, column_mapping=None):
    """
    Normalize column names to lowercase with underscores.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column_mapping (dict): Custom column name mapping
    
    Returns:
        pd.DataFrame: DataFrame with normalized column names
    """
    df_normalized = df.copy()
    
    if column_mapping:
        df_normalized = df_normalized.rename(columns=column_mapping)
    else:
        # Default normalization: lowercase, replace spaces with underscores
        new_columns = {}
        for col in df_normalized.columns:
            new_name = str(col).lower().replace(' ', '_').strip()
            new_columns[col] = new_name
        df_normalized = df_normalized.rename(columns=new_columns)
    
    return df_normalized

def remove_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Remove outliers from specified columns using IQR or Z-score method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to check for outliers
        method (str): 'iqr' for interquartile range or 'zscore' for standard deviations
        threshold (float): Threshold for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df_clean.columns or not pd.api.types.is_numeric_dtype(df_clean[col]):
            continue
            
        if method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        elif method == 'zscore':
            from scipy import stats
            z_scores = stats.zscore(df_clean[col].dropna())
            abs_z_scores = abs(z_scores)
            df_clean = df_clean[abs_z_scores < threshold]
    
    return df_clean