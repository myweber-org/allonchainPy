
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        factor: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        threshold: Z-score threshold (default 3)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    return (data[column] - min_val) / (max_val - min_val)

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    return (data[column] - mean_val) / std_val

def clean_dataset(data, numeric_columns=None, method='iqr', norm_method='minmax'):
    """
    Comprehensive data cleaning function.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to clean (default: all numeric)
        method: outlier removal method ('iqr' or 'zscore')
        norm_method: normalization method ('minmax' or 'zscore')
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        if method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
        elif method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, column)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        if norm_method == 'minmax':
            cleaned_data[f'{column}_normalized'] = normalize_minmax(cleaned_data, column)
        elif norm_method == 'zscore':
            cleaned_data[f'{column}_normalized'] = normalize_zscore(cleaned_data, column)
        else:
            raise ValueError(f"Unknown normalization method: {norm_method}")
    
    return cleaned_data

def get_summary_statistics(data):
    """
    Get summary statistics for numeric columns.
    
    Args:
        data: pandas DataFrame
    
    Returns:
        DataFrame with summary statistics
    """
    numeric_data = data.select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        return pd.DataFrame()
    
    summary = numeric_data.describe().T
    summary['skewness'] = numeric_data.skew()
    summary['kurtosis'] = numeric_data.kurtosis()
    summary['missing'] = numeric_data.isnull().sum()
    summary['missing_pct'] = (numeric_data.isnull().sum() / len(data)) * 100
    
    return summary

def detect_skewed_columns(data, threshold=0.5):
    """
    Detect columns with significant skewness.
    
    Args:
        data: pandas DataFrame
        threshold: absolute skewness threshold (default 0.5)
    
    Returns:
        List of skewed column names
    """
    numeric_data = data.select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        return []
    
    skewness = numeric_data.skew()
    skewed_columns = skewness[abs(skewness) > threshold].index.tolist()
    
    return skewed_columnsimport pandas as pd
import numpy as np

def clean_dataframe(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    missing_strategy (str): Strategy for handling missing values.
        Options: 'mean', 'median', 'mode', 'drop'.
    outlier_threshold (float): Number of standard deviations for outlier detection.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif missing_strategy in ['mean', 'median', 'mode']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if missing_strategy == 'mean':
                fill_value = cleaned_df[col].mean()
            elif missing_strategy == 'median':
                fill_value = cleaned_df[col].median()
            elif missing_strategy == 'mode':
                fill_value = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else np.nan
            cleaned_df[col] = cleaned_df[col].fillna(fill_value)
    
    # Handle outliers using z-score method
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
        cleaned_df = cleaned_df[z_scores < outlier_threshold]
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (is_valid, error_message)
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

# Example usage
if __name__ == "__main__":
    # Create sample data with missing values and outliers
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],  # Contains NaN and outlier
        'B': [5, 6, 7, np.nan, 9],
        'C': [10, 11, 12, 13, 14]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    # Clean the data
    cleaned = clean_dataframe(df, missing_strategy='mean', outlier_threshold=2)
    print("Cleaned DataFrame:")
    print(cleaned)
    print()
    
    # Validate the cleaned data
    is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
    print(f"Validation: {is_valid}")
    print(f"Message: {message}")
import pandas as pd
import numpy as np
from typing import Optional, List

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> 'DataCleaner':
        self.df = self.df.drop_duplicates(subset=subset)
        return self
        
    def handle_missing_values(self, 
                             strategy: str = 'mean',
                             columns: Optional[List[str]] = None) -> 'DataCleaner':
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns:
                if strategy == 'mean':
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif strategy == 'median':
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                elif strategy == 'mode':
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                elif strategy == 'drop':
                    self.df = self.df.dropna(subset=[col])
                    
        return self
        
    def remove_outliers(self, 
                       columns: Optional[List[str]] = None,
                       threshold: float = 3.0) -> 'DataCleaner':
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns:
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                self.df = self.df[z_scores < threshold]
                
        return self
        
    def normalize_data(self, 
                      columns: Optional[List[str]] = None,
                      method: str = 'minmax') -> 'DataCleaner':
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns:
                if method == 'minmax':
                    self.df[col] = (self.df[col] - self.df[col].min()) / (self.df[col].max() - self.df[col].min())
                elif method == 'zscore':
                    self.df[col] = (self.df[col] - self.df[col].mean()) / self.df[col].std()
                    
        return self
        
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df
        
    def get_cleaning_report(self) -> dict:
        return {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': self.df.shape[0],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'columns_removed': self.original_shape[1] - self.df.shape[1]
        }

def load_and_clean_csv(filepath: str, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(filepath, **kwargs)
    cleaner = DataCleaner(df)
    
    cleaner.handle_missing_values(strategy='mean')
    cleaner.remove_duplicates()
    cleaner.remove_outliers(threshold=3.0)
    cleaner.normalize_data(method='minmax')
    
    report = cleaner.get_cleaning_report()
    print(f"Data cleaning completed. Removed {report['rows_removed']} rows.")
    
    return cleaner.get_cleaned_data()