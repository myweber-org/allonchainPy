
import pandas as pd
import numpy as np
from typing import Union, List, Dict

def remove_outliers_iqr(
    df: pd.DataFrame,
    columns: Union[str, List[str]],
    multiplier: float = 1.5
) -> pd.DataFrame:
    """
    Remove outliers using the Interquartile Range method.
    
    Parameters:
    df: Input DataFrame
    columns: Column name or list of column names to process
    multiplier: IQR multiplier for outlier detection
    
    Returns:
    DataFrame with outliers removed
    """
    if isinstance(columns, str):
        columns = [columns]
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'mean',
    columns: Union[str, List[str], None] = None
) -> pd.DataFrame:
    """
    Handle missing values in specified columns.
    
    Parameters:
    df: Input DataFrame
    strategy: Imputation strategy ('mean', 'median', 'mode', 'drop')
    columns: Columns to process (None processes all columns)
    
    Returns:
    DataFrame with handled missing values
    """
    df_processed = df.copy()
    
    if columns is None:
        columns = df.columns.tolist()
    elif isinstance(columns, str):
        columns = [columns]
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if strategy == 'drop':
            df_processed = df_processed.dropna(subset=[col])
        elif strategy == 'mean':
            df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
        elif strategy == 'median':
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        elif strategy == 'mode':
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
    
    return df_processed.reset_index(drop=True)

def normalize_columns(
    df: pd.DataFrame,
    columns: Union[str, List[str]],
    method: str = 'minmax'
) -> pd.DataFrame:
    """
    Normalize specified columns.
    
    Parameters:
    df: Input DataFrame
    columns: Column name or list of column names to normalize
    method: Normalization method ('minmax' or 'zscore')
    
    Returns:
    DataFrame with normalized columns
    """
    if isinstance(columns, str):
        columns = [columns]
    
    df_normalized = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val != min_val:
                df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val != 0:
                df_normalized[col] = (df[col] - mean_val) / std_val
    
    return df_normalized

def get_data_summary(df: pd.DataFrame) -> Dict:
    """
    Generate a comprehensive summary of the DataFrame.
    
    Parameters:
    df: Input DataFrame
    
    Returns:
    Dictionary containing data summary
    """
    summary = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_stats': {},
        'categorical_stats': {}
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    
    for col in numeric_cols:
        summary['numeric_stats'][col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'skewness': df[col].skew()
        }
    
    for col in categorical_cols:
        summary['categorical_stats'][col] = {
            'unique_count': df[col].nunique(),
            'top_value': df[col].mode().iloc[0] if not df[col].mode().empty else None,
            'top_count': df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0
        }
    
    return summary

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate if DataFrame contains required columns and has valid data.
    
    Parameters:
    df: DataFrame to validate
    required_columns: List of required column names
    
    Returns:
    Boolean indicating if DataFrame is valid
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return False
    
    if df.empty:
        print("DataFrame is empty")
        return False
    
    return Trueimport numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns):
    """
    Remove outliers using the Interquartile Range method.
    Returns a cleaned DataFrame.
    """
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean.reset_index(drop=True)

def normalize_minmax(df, columns):
    """
    Normalize specified columns using Min-Max scaling.
    Returns DataFrame with normalized columns.
    """
    df_norm = df.copy()
    for col in columns:
        if col in df_norm.columns:
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            if max_val > min_val:
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
    return df_norm

def z_score_normalize(df, columns):
    """
    Normalize specified columns using Z-score normalization.
    Returns DataFrame with normalized columns.
    """
    df_z = df.copy()
    for col in columns:
        if col in df_z.columns:
            mean_val = df_z[col].mean()
            std_val = df_z[col].std()
            if std_val > 0:
                df_z[col] = (df_z[col] - mean_val) / std_val
    return df_z

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    Strategies: 'mean', 'median', 'mode', 'drop'
    """
    df_handled = df.copy()
    if columns is None:
        columns = df_handled.columns
    
    for col in columns:
        if col in df_handled.columns:
            if strategy == 'mean':
                fill_value = df_handled[col].mean()
            elif strategy == 'median':
                fill_value = df_handled[col].median()
            elif strategy == 'mode':
                fill_value = df_handled[col].mode()[0] if not df_handled[col].mode().empty else 0
            elif strategy == 'drop':
                df_handled = df_handled.dropna(subset=[col])
                continue
            else:
                raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")
            
            df_handled[col] = df_handled[col].fillna(fill_value)
    
    return df_handled.reset_index(drop=True)