
import pandas as pd
import numpy as np
from typing import Union, List, Dict

def remove_duplicates(df: pd.DataFrame, subset: Union[List[str], None] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean', columns: Union[List[str], None] = None) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: 'mean', 'median', 'mode', or 'drop'
        columns: Specific columns to process
    
    Returns:
        DataFrame with handled missing values
    """
    df_copy = df.copy()
    
    if columns is None:
        columns = df_copy.columns
    
    for col in columns:
        if df_copy[col].dtype in ['int64', 'float64']:
            if strategy == 'mean':
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)
            elif strategy == 'median':
                df_copy[col].fillna(df_copy[col].median(), inplace=True)
            elif strategy == 'mode':
                df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                df_copy = df_copy.dropna(subset=[col])
        else:
            if strategy == 'mode':
                df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                df_copy = df_copy.dropna(subset=[col])
    
    return df_copy

def normalize_columns(df: pd.DataFrame, columns: Union[List[str], None] = None) -> pd.DataFrame:
    """
    Normalize numerical columns to range [0, 1].
    
    Args:
        df: Input DataFrame
        columns: Specific columns to normalize
    
    Returns:
        DataFrame with normalized columns
    """
    df_copy = df.copy()
    
    if columns is None:
        columns = [col for col in df_copy.columns if df_copy[col].dtype in ['int64', 'float64']]
    
    for col in columns:
        if df_copy[col].dtype in ['int64', 'float64']:
            col_min = df_copy[col].min()
            col_max = df_copy[col].max()
            if col_max != col_min:
                df_copy[col] = (df_copy[col] - col_min) / (col_max - col_min)
    
    return df_copy

def detect_outliers_iqr(df: pd.DataFrame, columns: Union[List[str], None] = None) -> Dict[str, List[int]]:
    """
    Detect outliers using IQR method.
    
    Args:
        df: Input DataFrame
        columns: Specific columns to check
    
    Returns:
        Dictionary with column names as keys and outlier indices as values
    """
    if columns is None:
        columns = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
    
    outliers = {}
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
        if outlier_indices:
            outliers[col] = outlier_indices
    
    return outliers

def clean_dataset(df: pd.DataFrame, 
                  remove_dups: bool = True,
                  handle_nulls: Union[str, None] = 'mean',
                  normalize: bool = False) -> pd.DataFrame:
    """
    Comprehensive dataset cleaning pipeline.
    
    Args:
        df: Input DataFrame
        remove_dups: Whether to remove duplicates
        handle_nulls: Strategy for handling nulls or None to skip
        normalize: Whether to normalize numerical columns
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if remove_dups:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if handle_nulls:
        cleaned_df = handle_missing_values(cleaned_df, strategy=handle_nulls)
    
    if normalize:
        cleaned_df = normalize_columns(cleaned_df)
    
    return cleaned_df
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
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
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            elif fill_missing == 'median':
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None, inplace=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (bool, str) indicating success and message.
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame validation passed"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 30, 40, 50],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B'])
    print(f"\nValidation: {is_valid}, Message: {message}")import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': len(df[column])
    }
    
    return stats

def clean_dataset(df, columns_to_clean):
    """
    Clean multiple columns in a dataset by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary of statistics for each cleaned column
    """
    cleaned_df = df.copy()
    statistics = {}
    
    for column in columns_to_clean:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            
            stats = calculate_summary_statistics(cleaned_df, column)
            stats['outliers_removed'] = removed_count
            statistics[column] = stats
    
    return cleaned_df, statistics

if __name__ == "__main__":
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 27, 28, 29, 30, 100],
        'humidity': [45, 46, 47, 48, 49, 50, 51, 52, 53, 150],
        'pressure': [1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 2000]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    columns_to_clean = ['temperature', 'humidity', 'pressure']
    cleaned_df, stats = clean_dataset(df, columns_to_clean)
    
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    print("Summary Statistics:")
    for column, column_stats in stats.items():
        print(f"\n{column}:")
        for stat_name, stat_value in column_stats.items():
            print(f"  {stat_name}: {stat_value}")
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def calculate_summary_statistics(df):
    """
    Calculate summary statistics for numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return {}
    
    stats = {}
    for col in numeric_cols:
        stats[col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'count': df[col].count()
        }
    
    return stats

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.Series: Normalized column values
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val == min_val:
            return pd.Series([0.5] * len(df), index=df.index)
        return (df[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val == 0:
            return pd.Series([0] * len(df), index=df.index)
        return (df[column] - mean_val) / std_val
    
    else:
        raise ValueError("Method must be either 'minmax' or 'zscore'")

def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        return df.dropna(subset=numeric_cols)
    
    df_filled = df.copy()
    
    for col in numeric_cols:
        if df[col].isnull().any():
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
            else:
                raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")
            
            df_filled[col] = df[col].fillna(fill_value)
    
    return df_filled

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
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"