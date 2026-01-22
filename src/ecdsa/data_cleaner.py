
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, columns=None):
    """
    Remove outliers from specified columns using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to process. If None, processes all numeric columns.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize specified columns in DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to normalize
    method (str): Normalization method - 'minmax' or 'zscore'
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_norm = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val != min_val:
                df_norm[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df_norm[col] = 0
                
        elif method == 'zscore':
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val != 0:
                df_norm[col] = (df[col] - mean_val) / std_val
            else:
                df_norm[col] = 0
    
    return df_norm

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Strategy for handling missing values - 'mean', 'median', 'mode', or 'drop'
    columns (list): List of column names to process
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    if columns is None:
        columns = df.columns.tolist()
    
    df_processed = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if strategy == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
            df_processed[col] = df[col].fillna(df[col].mean())
        elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[col]):
            df_processed[col] = df[col].fillna(df[col].median())
        elif strategy == 'mode':
            df_processed[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None)
        elif strategy == 'drop':
            df_processed = df_processed.dropna(subset=[col])
    
    return df_processed.reset_index(drop=True)
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        multiplier: IQR multiplier (default 1.5)
    
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
        DataFrame with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        data[f'{column}_normalized'] = 0.5
    else:
        data[f'{column}_normalized'] = (data[column] - min_val) / (max_val - min_val)
    
    return data

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        DataFrame with standardized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        data[f'{column}_standardized'] = 0
    else:
        data[f'{column}_standardized'] = (data[column] - mean_val) / std_val
    
    return data

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Args:
        data: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', 'drop')
        columns: list of columns to process (None for all numeric columns)
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    data_processed = data.copy()
    
    for col in columns:
        if col not in data.columns:
            continue
            
        if strategy == 'drop':
            data_processed = data_processed.dropna(subset=[col])
        elif strategy == 'mean':
            data_processed[col] = data_processed[col].fillna(data_processed[col].mean())
        elif strategy == 'median':
            data_processed[col] = data_processed[col].fillna(data_processed[col].median())
        elif strategy == 'mode':
            data_processed[col] = data_processed[col].fillna(data_processed[col].mode()[0])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return data_processed

def create_data_summary(data):
    """
    Create a summary statistics DataFrame.
    
    Args:
        data: pandas DataFrame
    
    Returns:
        DataFrame with summary statistics
    """
    summary = pd.DataFrame({
        'column': data.columns,
        'dtype': data.dtypes.values,
        'non_null_count': data.count().values,
        'null_count': data.isnull().sum().values,
        'null_percentage': (data.isnull().sum() / len(data) * 100).values,
        'unique_count': data.nunique().values
    })
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        numeric_stats = data[numeric_cols].describe().T
        summary = summary.merge(numeric_stats, left_on='column', right_index=True, how='left')
    
    return summary