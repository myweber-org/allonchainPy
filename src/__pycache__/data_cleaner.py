
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns
        
    def remove_outliers_iqr(self, columns=None, threshold=1.5):
        if columns is None:
            columns = self.numeric_columns
            
        df_clean = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        return df_clean
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.numeric_columns
            
        df_clean = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                z_scores = np.abs(stats.zscore(df_clean[col]))
                df_clean = df_clean[z_scores < threshold]
        return df_clean
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
            
        df_normalized = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                min_val = df_normalized[col].min()
                max_val = df_normalized[col].max()
                if max_val != min_val:
                    df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
        return df_normalized
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
            
        df_normalized = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                mean_val = df_normalized[col].mean()
                std_val = df_normalized[col].std()
                if std_val > 0:
                    df_normalized[col] = (df_normalized[col] - mean_val) / std_val
        return df_normalized
    
    def fill_missing_mean(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
            
        df_filled = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
        return df_filled
    
    def get_summary(self):
        summary = {
            'original_shape': self.df.shape,
            'numeric_columns': list(self.numeric_columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict()
        }
        return summary
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_summary_stats(data, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    stats = {
        'mean': np.mean(data[column]),
        'median': np.median(data[column]),
        'std': np.std(data[column]),
        'min': np.min(data[column]),
        'max': np.max(data[column]),
        'count': len(data[column])
    }
    return stats

def process_numerical_column(data, column):
    """
    Complete processing pipeline for numerical column.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    tuple: (cleaned_data, original_stats, cleaned_stats)
    """
    original_stats = calculate_summary_stats(data, column)
    cleaned_data = remove_outliers_iqr(data, column)
    cleaned_stats = calculate_summary_stats(cleaned_data, column)
    
    return cleaned_data, original_stats, cleaned_stats
import re
import pandas as pd
from typing import Optional, List, Dict, Any

def remove_special_characters(text: str, keep_chars: str = '') -> str:
    pattern = f'[^a-zA-Z0-9\\s{re.escape(keep_chars)}]'
    return re.sub(pattern, '', text)

def validate_email(email: str) -> bool:
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def normalize_phone_number(phone: str, country_code: str = '+1') -> Optional[str]:
    digits = re.sub(r'\D', '', phone)
    if not digits:
        return None
    if not digits.startswith('1'):
        digits = '1' + digits
    return f'{country_code}{digits[1:]}'

def clean_dataframe(df: pd.DataFrame, 
                    drop_na_threshold: float = 0.5,
                    date_columns: List[str] = None) -> pd.DataFrame:
    cleaned_df = df.copy()
    
    if date_columns:
        for col in date_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
    
    na_percentage = cleaned_df.isna().sum() / len(cleaned_df)
    columns_to_drop = na_percentage[na_percentage > drop_na_threshold].index
    cleaned_df = cleaned_df.drop(columns=columns_to_drop)
    
    cleaned_df = cleaned_df.dropna(thresh=len(cleaned_df.columns) * 0.7)
    
    return cleaned_df

def generate_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isna().sum().sum(),
        'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        'date_columns': df.select_dtypes(include=['datetime']).columns.tolist()
    }
    
    if summary['numeric_columns']:
        summary['numeric_stats'] = df[summary['numeric_columns']].describe().to_dict()
    
    return summary

def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return (series < lower_bound) | (series > upper_bound)

def format_column_names(df: pd.DataFrame) -> pd.DataFrame:
    formatted_df = df.copy()
    formatted_df.columns = (
        formatted_df.columns
        .str.lower()
        .str.replace(r'[^\w\s]', '', regex=True)
        .str.replace(r'\s+', '_', regex=True)
        .str.strip('_')
    )
    return formatted_df
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
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

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If columns is None, clean all numeric columns.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): List of column names to clean.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[col]):
            original_len = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_len - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{col}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'id': range(1, 101),
        'value': np.concatenate([
            np.random.normal(100, 10, 90),
            np.random.normal(300, 50, 10)
        ]),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Original data shape: {df.shape}")
    
    cleaned_df = clean_numeric_data(df, columns=['value'])
    print(f"Cleaned data shape: {cleaned_df.shape}")
    print(f"Outliers removed: {len(df) - len(cleaned_df)}")