import pandas as pd
import numpy as np
from scipy import stats

def clean_dataset(df, numeric_columns=None, method='median', z_threshold=3):
    """
    Clean dataset by handling missing values and removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    numeric_columns (list): List of numeric column names to process
    method (str): Imputation method ('mean', 'median', 'mode')
    z_threshold (float): Z-score threshold for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    df_clean = df.copy()
    
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_columns:
        if col not in df_clean.columns:
            continue
            
        col_data = df_clean[col]
        
        missing_mask = col_data.isnull()
        if missing_mask.any():
            if method == 'mean':
                fill_value = col_data.mean()
            elif method == 'median':
                fill_value = col_data.median()
            elif method == 'mode':
                fill_value = col_data.mode()[0] if not col_data.mode().empty else 0
            else:
                fill_value = 0
                
            df_clean.loc[missing_mask, col] = fill_value
        
        if len(col_data) > 10:
            z_scores = np.abs(stats.zscore(df_clean[col]))
            outlier_mask = z_scores > z_threshold
            
            if outlier_mask.any():
                median_val = df_clean[col].median()
                df_clean.loc[outlier_mask, col] = median_val
    
    return df_clean

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate dataframe structure and content.
    
    Parameters:
    df (pd.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

def get_summary_statistics(df):
    """
    Generate summary statistics for numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    
    Returns:
    pd.DataFrame: Summary statistics
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return pd.DataFrame()
    
    stats_dict = {
        'count': numeric_df.count(),
        'mean': numeric_df.mean(),
        'std': numeric_df.std(),
        'min': numeric_df.min(),
        '25%': numeric_df.quantile(0.25),
        '50%': numeric_df.quantile(0.50),
        '75%': numeric_df.quantile(0.75),
        'max': numeric_df.max(),
        'missing': numeric_df.isnull().sum()
    }
    
    return pd.DataFrame(stats_dict)
def deduplicate_list(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result