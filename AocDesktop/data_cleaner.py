
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_duplicates (bool): Whether to remove duplicate rows
    fill_missing (bool): Whether to fill missing values
    fill_value: Value to use for filling missing data
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
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
    
    return True, "DataFrame is valid"
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    multiplier (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df.copy()

def normalize_minmax(dataframe, columns=None):
    """
    Normalize specified columns using min-max scaling.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    columns (list): List of column names to normalize. If None, normalize all numeric columns.
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    df_copy = dataframe.copy()
    
    if columns is None:
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for col in columns:
        if col not in df_copy.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        
        if not np.issubdtype(df_copy[col].dtype, np.number):
            raise TypeError(f"Column '{col}' must be numeric for normalization")
        
        col_min = df_copy[col].min()
        col_max = df_copy[col].max()
        
        if col_max != col_min:
            df_copy[col] = (df_copy[col] - col_min) / (col_max - col_min)
        else:
            df_copy[col] = 0
    
    return df_copy

def standardize_zscore(dataframe, columns=None):
    """
    Standardize specified columns using z-score normalization.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    columns (list): List of column names to standardize. If None, standardize all numeric columns.
    
    Returns:
    pd.DataFrame: DataFrame with standardized columns
    """
    df_copy = dataframe.copy()
    
    if columns is None:
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for col in columns:
        if col not in df_copy.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        
        if not np.issubdtype(df_copy[col].dtype, np.number):
            raise TypeError(f"Column '{col}' must be numeric for standardization")
        
        col_mean = df_copy[col].mean()
        col_std = df_copy[col].std()
        
        if col_std > 0:
            df_copy[col] = (df_copy[col] - col_mean) / col_std
        else:
            df_copy[col] = 0
    
    return df_copy

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    strategy (str): Imputation strategy ('mean', 'median', 'mode', 'drop')
    columns (list): List of column names to process. If None, process all columns.
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    df_copy = dataframe.copy()
    
    if columns is None:
        columns = df_copy.columns
    
    for col in columns:
        if col not in df_copy.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        
        if df_copy[col].isnull().any():
            if strategy == 'drop':
                df_copy = df_copy.dropna(subset=[col])
            elif strategy == 'mean':
                if np.issubdtype(df_copy[col].dtype, np.number):
                    df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
            elif strategy == 'median':
                if np.issubdtype(df_copy[col].dtype, np.number):
                    df_copy[col] = df_copy[col].fillna(df_copy[col].median())
            elif strategy == 'mode':
                mode_value = df_copy[col].mode()
                if not mode_value.empty:
                    df_copy[col] = df_copy[col].fillna(mode_value.iloc[0])
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
    
    return df_copy

def clean_dataset(dataframe, outlier_columns=None, normalize=True, standardize=False, missing_strategy='mean'):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    outlier_columns (list): Columns to apply outlier removal
    normalize (bool): Whether to apply min-max normalization
    standardize (bool): Whether to apply z-score standardization
    missing_strategy (str): Strategy for handling missing values
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    df_clean = dataframe.copy()
    
    df_clean = handle_missing_values(df_clean, strategy=missing_strategy)
    
    if outlier_columns:
        for col in outlier_columns:
            if col in df_clean.columns:
                df_clean = remove_outliers_iqr(df_clean, col)
    
    if normalize:
        df_clean = normalize_minmax(df_clean)
    
    if standardize:
        df_clean = standardize_zscore(df_clean)
    
    return df_clean