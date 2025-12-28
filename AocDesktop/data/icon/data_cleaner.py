
import pandas as pd
import numpy as np
from typing import Union, List, Dict

def remove_duplicates(df: pd.DataFrame, subset: Union[List[str], str] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Column(s) to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def handle_missing_values(df: pd.DataFrame, 
                         strategy: str = 'mean',
                         columns: List[str] = None) -> pd.DataFrame:
    """
    Handle missing values in DataFrame columns.
    
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
        if df_copy[col].isnull().any():
            if strategy == 'mean':
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)
            elif strategy == 'median':
                df_copy[col].fillna(df_copy[col].median(), inplace=True)
            elif strategy == 'mode':
                df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                df_copy = df_copy.dropna(subset=[col])
    
    return df_copy

def normalize_column(df: pd.DataFrame, 
                    column: str,
                    method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize a column using specified method.
    
    Args:
        df: Input DataFrame
        column: Column to normalize
        method: 'minmax' or 'zscore'
    
    Returns:
        DataFrame with normalized column
    """
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        df_copy[column] = (df_copy[column] - mean_val) / std_val
    
    return df_copy

def detect_outliers_iqr(df: pd.DataFrame, 
                       column: str,
                       threshold: float = 1.5) -> Dict:
    """
    Detect outliers using Interquartile Range method.
    
    Args:
        df: Input DataFrame
        column: Column to analyze
        threshold: IQR multiplier threshold
    
    Returns:
        Dictionary with outlier statistics
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    return {
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'outlier_count': len(outliers),
        'outlier_percentage': (len(outliers) / len(df)) * 100,
        'outliers': outliers
    }

def clean_data_pipeline(df: pd.DataFrame,
                       steps: List[Dict]) -> pd.DataFrame:
    """
    Execute a series of data cleaning steps.
    
    Args:
        df: Input DataFrame
        steps: List of cleaning step configurations
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for step in steps:
        action = step.get('action')
        
        if action == 'remove_duplicates':
            subset = step.get('subset')
            cleaned_df = remove_duplicates(cleaned_df, subset)
        
        elif action == 'handle_missing':
            strategy = step.get('strategy', 'mean')
            columns = step.get('columns')
            cleaned_df = handle_missing_values(cleaned_df, strategy, columns)
        
        elif action == 'normalize':
            column = step.get('column')
            method = step.get('method', 'minmax')
            cleaned_df = normalize_column(cleaned_df, column, method)
    
    return cleaned_df
import pandas as pd
import numpy as np

def remove_outliers(df, column, threshold=3):
    mean = df[column].mean()
    std = df[column].std()
    z_scores = np.abs((df[column] - mean) / std)
    return df[z_scores < threshold]

def normalize_column(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path):
    df = pd.read_csv(file_path)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers(df, col)
        df = normalize_column(df, col)
    
    return df

if __name__ == "__main__":
    cleaned_df = clean_dataset('data.csv')
    cleaned_df.to_csv('cleaned_data.csv', index=False)