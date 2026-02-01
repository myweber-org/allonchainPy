
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns):
    cleaned_df = df.copy()
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            cleaned_df = cleaned_df[mask]
    return cleaned_df.reset_index(drop=True)

def remove_outliers_zscore(df, columns, threshold=3):
    cleaned_df = df.copy()
    for col in columns:
        if col in df.columns:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            mask = z_scores < threshold
            valid_indices = df[col].dropna().index[mask]
            cleaned_df = cleaned_df.loc[valid_indices.union(df[df[col].isna()].index)]
    return cleaned_df.reset_index(drop=True)

def normalize_minmax(df, columns):
    normalized_df = df.copy()
    for col in columns:
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max != col_min:
                normalized_df[col] = (df[col] - col_min) / (col_max - col_min)
            else:
                normalized_df[col] = 0
    return normalized_df

def normalize_zscore(df, columns):
    normalized_df = df.copy()
    for col in columns:
        if col in df.columns:
            col_mean = df[col].mean()
            col_std = df[col].std()
            if col_std > 0:
                normalized_df[col] = (df[col] - col_mean) / col_std
            else:
                normalized_df[col] = 0
    return normalized_df

def handle_missing_values(df, strategy='mean', columns=None):
    processed_df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
            elif strategy == 'zero':
                fill_value = 0
            else:
                fill_value = df[col].mean()
            
            processed_df[col] = df[col].fillna(fill_value)
    
    return processed_df

def get_data_summary(df):
    summary = {
        'original_shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
    }
    return summary

def clean_dataset(df, numeric_columns=None, outlier_method='iqr', normalize_method=None, missing_strategy='mean'):
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy, columns=numeric_columns)
    
    if outlier_method == 'iqr':
        cleaned_df = remove_outliers_iqr(cleaned_df, numeric_columns)
    elif outlier_method == 'zscore':
        cleaned_df = remove_outliers_zscore(cleaned_df, numeric_columns)
    
    if normalize_method == 'minmax':
        cleaned_df = normalize_minmax(cleaned_df, numeric_columns)
    elif normalize_method == 'zscore':
        cleaned_df = normalize_zscore(cleaned_df, numeric_columns)
    
    return cleaned_dfimport pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (str or dict): Method to fill missing values. 
            Options: 'mean', 'median', 'mode', or a dictionary of column:value pairs.
            If None, missing values are not filled.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        elif fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
        min_rows (int): Minimum number of rows required.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"