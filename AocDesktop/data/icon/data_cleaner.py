
import re
import json
from typing import Dict, Any, Optional, List

def sanitize_string(input_string: str) -> str:
    """Remove extra whitespace and normalize line endings."""
    if not isinstance(input_string, str):
        raise TypeError("Input must be a string")
    cleaned = re.sub(r'\s+', ' ', input_string.strip())
    return cleaned.replace('\r\n', '\n').replace('\r', '\n')

def validate_email(email: str) -> bool:
    """Check if the provided string is a valid email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def parse_json_safe(json_string: str) -> Optional[Dict[str, Any]]:
    """Safely parse a JSON string, returning None on failure."""
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        return None

def filter_list_by_prefix(items: List[str], prefix: str) -> List[str]:
    """Filter a list of strings, keeping only those starting with a prefix."""
    return [item for item in items if item.startswith(prefix)]

def calculate_checksum(data: str) -> str:
    """Calculate a simple checksum for a string."""
    if not data:
        return '0'
    hash_val = 0
    for char in data:
        hash_val = (hash_val * 31 + ord(char)) & 0xFFFFFFFF
    return format(hash_val, '08x')
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (bool): Whether to fill missing values.
    fill_value: Value to use for filling missing data.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
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
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    return True, "DataFrame is valid"

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to process.
    method (str): 'iqr' for interquartile range or 'zscore' for standard deviation.
    threshold (float): Threshold for outlier detection.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    data = df[column].copy()
    
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (data >= lower_bound) & (data <= upper_bound)
    
    elif method == 'zscore':
        mean = data.mean()
        std = data.std()
        z_scores = np.abs((data - mean) / std)
        mask = z_scores <= threshold
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return df[mask]

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to normalize.
    method (str): Normalization method ('minmax' or 'zscore').
    
    Returns:
    pd.DataFrame: DataFrame with normalized column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        if max_val != min_val:
            df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val != 0:
            df_copy[column] = (df_copy[column] - mean_val) / std_val
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df_copy

def sample_dataframe(df, sample_size=1000, random_state=42):
    """
    Sample rows from DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    sample_size (int): Number of rows to sample.
    random_state (int): Random seed for reproducibility.
    
    Returns:
    pd.DataFrame: Sampled DataFrame.
    """
    if len(df) <= sample_size:
        return df
    
    return df.sample(n=sample_size, random_state=random_state)
import pandas as pd
import numpy as np
from typing import List, Optional

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape

    def remove_duplicates(self, subset: Optional[List[str]] = None) -> 'DataCleaner':
        self.df = self.df.drop_duplicates(subset=subset, keep='first')
        return self

    def normalize_column(self, column_name: str, method: str = 'minmax') -> 'DataCleaner':
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame")

        if method == 'minmax':
            col_min = self.df[column_name].min()
            col_max = self.df[column_name].max()
            if col_max != col_min:
                self.df[column_name] = (self.df[column_name] - col_min) / (col_max - col_min)
        elif method == 'zscore':
            col_mean = self.df[column_name].mean()
            col_std = self.df[column_name].std()
            if col_std > 0:
                self.df[column_name] = (self.df[column_name] - col_mean) / col_std
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return self

    def fill_missing(self, column_name: str, strategy: str = 'mean') -> 'DataCleaner':
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame")

        if strategy == 'mean':
            fill_value = self.df[column_name].mean()
        elif strategy == 'median':
            fill_value = self.df[column_name].median()
        elif strategy == 'mode':
            fill_value = self.df[column_name].mode().iloc[0] if not self.df[column_name].mode().empty else np.nan
        elif strategy == 'zero':
            fill_value = 0
        else:
            raise ValueError(f"Unknown fill strategy: {strategy}")

        self.df[column_name] = self.df[column_name].fillna(fill_value)
        return self

    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df.copy()

    def get_stats(self) -> dict:
        cleaned_shape = self.df.shape
        return {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': cleaned_shape[0],
            'cleaned_columns': cleaned_shape[1],
            'rows_removed': self.original_shape[0] - cleaned_shape[0]
        }

def clean_dataset(df: pd.DataFrame, 
                  duplicate_columns: Optional[List[str]] = None,
                  normalize_columns: Optional[List[str]] = None,
                  fill_columns: Optional[List[str]] = None) -> pd.DataFrame:
    cleaner = DataCleaner(df)
    
    if duplicate_columns:
        cleaner.remove_duplicates(duplicate_columns)
    
    if normalize_columns:
        for col in normalize_columns:
            cleaner.normalize_column(col)
    
    if fill_columns:
        for col in fill_columns:
            cleaner.fill_missing(col)
    
    return cleaner.get_cleaned_data()import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns, factor=1.5):
    """
    Remove outliers using IQR method
    """
    df_clean = df.copy()
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def remove_outliers_zscore(df, columns, threshold=3):
    """
    Remove outliers using Z-score method
    """
    df_clean = df.copy()
    for col in columns:
        if col in df.columns:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            valid_indices = np.where(z_scores < threshold)[0]
            df_clean = df_clean.iloc[valid_indices]
    return df_clean

def normalize_minmax(df, columns):
    """
    Normalize data using min-max scaling
    """
    df_normalized = df.copy()
    for col in columns:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
    return df_normalized

def normalize_zscore(df, columns):
    """
    Normalize data using Z-score standardization
    """
    df_normalized = df.copy()
    for col in columns:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                df_normalized[col] = (df[col] - mean_val) / std_val
    return df_normalized

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    """
    Main cleaning pipeline
    """
    if outlier_method == 'iqr':
        df_clean = remove_outliers_iqr(df, numeric_columns)
    elif outlier_method == 'zscore':
        df_clean = remove_outliers_zscore(df, numeric_columns)
    else:
        df_clean = df.copy()
    
    if normalize_method == 'minmax':
        df_final = normalize_minmax(df_clean, numeric_columns)
    elif normalize_method == 'zscore':
        df_final = normalize_zscore(df_clean, numeric_columns)
    else:
        df_final = df_clean
    
    return df_final

def get_summary_statistics(df):
    """
    Generate summary statistics for the dataset
    """
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
    }
    
    if summary['numeric_columns']:
        numeric_stats = df[summary['numeric_columns']].describe().to_dict()
        summary['numeric_stats'] = numeric_stats
    
    return summary
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    mask = z_scores < threshold
    
    filtered_data = data[mask]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].copy()
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].copy()
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, outlier_method='iqr', normalize_method='minmax', outlier_columns=None, normalize_columns=None):
    """
    Comprehensive data cleaning pipeline
    """
    cleaned_data = data.copy()
    
    if outlier_columns is None:
        outlier_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if normalize_columns is None:
        normalize_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    removal_stats = {}
    
    for column in outlier_columns:
        if column in data.columns and pd.api.types.is_numeric_dtype(data[column]):
            try:
                if outlier_method == 'iqr':
                    cleaned_data, removed = remove_outliers_iqr(cleaned_data, column)
                elif outlier_method == 'zscore':
                    cleaned_data, removed = remove_outliers_zscore(cleaned_data, column)
                else:
                    continue
                
                removal_stats[column] = removed
            except Exception as e:
                print(f"Error processing column {column}: {e}")
                continue
    
    for column in normalize_columns:
        if column in cleaned_data.columns and pd.api.types.is_numeric_dtype(cleaned_data[column]):
            try:
                if normalize_method == 'minmax':
                    cleaned_data[column] = normalize_minmax(cleaned_data, column)
                elif normalize_method == 'zscore':
                    cleaned_data[column] = normalize_zscore(cleaned_data, column)
            except Exception as e:
                print(f"Error normalizing column {column}: {e}")
                continue
    
    return cleaned_data, removal_stats

def validate_data(data, required_columns=None, allow_nan=False, min_rows=1):
    """
    Validate dataset structure and content
    """
    if len(data) < min_rows:
        raise ValueError(f"Dataset must have at least {min_rows} rows")
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if not allow_nan:
        nan_columns = data.columns[data.isnull().any()].tolist()
        if nan_columns:
            raise ValueError(f"NaN values found in columns: {nan_columns}")
    
    return True

def get_data_summary(data):
    """
    Generate comprehensive data summary
    """
    summary = {
        'shape': data.shape,
        'dtypes': data.dtypes.to_dict(),
        'missing_values': data.isnull().sum().to_dict(),
        'numeric_stats': {},
        'categorical_stats': {}
    }
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['numeric_stats'][col] = {
            'mean': data[col].mean(),
            'std': data[col].std(),
            'min': data[col].min(),
            'max': data[col].max(),
            'median': data[col].median()
        }
    
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        summary['categorical_stats'][col] = {
            'unique_count': data[col].nunique(),
            'top_value': data[col].mode().iloc[0] if not data[col].mode().empty else None,
            'top_count': data[col].value_counts().iloc[0] if not data[col].value_counts().empty else 0
        }
    
    return summary