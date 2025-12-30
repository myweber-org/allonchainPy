
import pandas as pd
import numpy as np
from typing import Optional

def clean_csv_data(
    filepath: str,
    missing_strategy: str = 'drop',
    fill_value: Optional[float] = None,
    remove_duplicates: bool = True
) -> pd.DataFrame:
    """
    Load and clean CSV data with configurable missing value handling.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    missing_strategy : str
        Strategy for handling missing values: 'drop', 'fill', or 'ignore'
    fill_value : float, optional
        Value to fill missing entries when strategy is 'fill'
    remove_duplicates : bool
        Whether to remove duplicate rows
    
    Returns:
    --------
    pd.DataFrame
        Cleaned DataFrame
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    
    original_shape = df.shape
    
    if remove_duplicates:
        df = df.drop_duplicates()
    
    if missing_strategy == 'drop':
        df = df.dropna()
    elif missing_strategy == 'fill':
        if fill_value is not None:
            df = df.fillna(fill_value)
        else:
            df = df.fillna(df.mean(numeric_only=True))
    elif missing_strategy == 'ignore':
        pass
    else:
        raise ValueError("missing_strategy must be 'drop', 'fill', or 'ignore'")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        df[numeric_cols] = df[numeric_cols].apply(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x
        )
    
    print(f"Original shape: {original_shape}")
    print(f"Cleaned shape: {df.shape}")
    print(f"Removed rows: {original_shape[0] - df.shape[0]}")
    
    return df

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate DataFrame for common data quality issues.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate
    
    Returns:
    --------
    bool
        True if DataFrame passes validation
    """
    if df.empty:
        print("Warning: DataFrame is empty")
        return False
    
    if df.isnull().any().any():
        print("Warning: DataFrame contains missing values")
        return False
    
    if df.duplicated().any():
        print("Warning: DataFrame contains duplicates")
        return False
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].std() == 0:
            print(f"Warning: Column '{col}' has zero variance")
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [5, 6, 7, np.nan, 9],
        'C': [10, 11, 12, 13, 10]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned = clean_csv_data(
        'test_data.csv',
        missing_strategy='fill',
        remove_duplicates=True
    )
    
    print("\nValidation result:", validate_dataframe(cleaned))
    print("\nCleaned DataFrame:")
    print(cleaned)
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, factor=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_clean = self.df.copy()
        for col in columns:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        self.df = df_clean
        return self
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_clean = self.df.copy()
        for col in columns:
            if col in df_clean.columns:
                z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                df_clean = df_clean[(z_scores < threshold) | df_clean[col].isna()]
        
        self.df = df_clean
        return self
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_normalized = self.df.copy()
        for col in columns:
            if col in df_normalized.columns:
                col_min = df_normalized[col].min()
                col_max = df_normalized[col].max()
                if col_max != col_min:
                    df_normalized[col] = (df_normalized[col] - col_min) / (col_max - col_min)
        
        self.df = df_normalized
        return self
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_normalized = self.df.copy()
        for col in columns:
            if col in df_normalized.columns:
                col_mean = df_normalized[col].mean()
                col_std = df_normalized[col].std()
                if col_std > 0:
                    df_normalized[col] = (df_normalized[col] - col_mean) / col_std
        
        self.df = df_normalized
        return self
    
    def fill_missing_mean(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_filled = self.df.copy()
        for col in columns:
            if col in df_filled.columns:
                df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
        
        self.df = df_filled
        return self
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_filled = self.df.copy()
        for col in columns:
            if col in df_filled.columns:
                df_filled[col] = df_filled[col].fillna(df_filled[col].median())
        
        self.df = df_filled
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'cleaned_rows': self.df.shape[0],
            'rows_removed': self.get_removed_count(),
            'original_columns': self.original_shape[1],
            'cleaned_columns': self.df.shape[1],
            'missing_values': self.df.isnull().sum().sum()
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 1, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    
    indices = np.random.choice(df.index, size=50, replace=False)
    df.loc[indices, 'feature_a'] = np.random.uniform(200, 300, 50)
    
    indices = np.random.choice(df.index, size=30, replace=False)
    df.loc[indices, 'feature_b'] = np.nan
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    print("Original data shape:", sample_df.shape)
    print("Missing values:", sample_df.isnull().sum().sum())
    
    cleaner = DataCleaner(sample_df)
    
    cleaner.remove_outliers_iqr(['feature_a', 'feature_b'])
    cleaner.fill_missing_median(['feature_b'])
    cleaner.normalize_minmax(['feature_a', 'feature_b', 'feature_c'])
    
    cleaned_df = cleaner.get_cleaned_data()
    summary = cleaner.get_summary()
    
    print("\nCleaned data shape:", cleaned_df.shape)
    print("Rows removed:", summary['rows_removed'])
    print("Missing values after cleaning:", summary['missing_values'])
    print("\nFirst 5 rows of cleaned data:")
    print(cleaned_df.head())
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

def z_score_normalize(data, column):
    """
    Normalize data using z-score normalization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column]
    
    normalized = (data[column] - mean_val) / std_val
    return normalized

def min_max_normalize(data, column, feature_range=(0, 1)):
    """
    Normalize data using min-max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column]
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    
    if feature_range != (0, 1):
        new_min, new_max = feature_range
        normalized = normalized * (new_max - new_min) + new_min
    
    return normalized

def detect_skewed_columns(data, threshold=0.5):
    """
    Detect columns with skewed distributions
    """
    skewed_columns = []
    
    for column in data.select_dtypes(include=[np.number]).columns:
        skewness = stats.skew(data[column].dropna())
        if abs(skewness) > threshold:
            skewed_columns.append((column, skewness))
    
    return sorted(skewed_columns, key=lambda x: abs(x[1]), reverse=True)

def log_transform(data, column):
    """
    Apply log transformation to reduce skewness
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if data[column].min() <= 0:
        transformed = np.log1p(data[column] - data[column].min() + 1)
    else:
        transformed = np.log(data[column])
    
    return transformed

def clean_dataset(data, numeric_columns=None, outlier_factor=1.5, normalize_method='zscore'):
    """
    Comprehensive data cleaning pipeline
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    cleaning_report = {}
    
    for column in numeric_columns:
        if column not in data.columns:
            continue
        
        original_count = len(cleaned_data)
        
        cleaned_data, removed = remove_outliers_iqr(cleaned_data, column, outlier_factor)
        cleaning_report[column] = {
            'outliers_removed': removed,
            'percentage_removed': (removed / original_count) * 100
        }
        
        if normalize_method == 'zscore':
            cleaned_data[f'{column}_normalized'] = z_score_normalize(cleaned_data, column)
        elif normalize_method == 'minmax':
            cleaned_data[f'{column}_normalized'] = min_max_normalize(cleaned_data, column)
    
    skewed_cols = detect_skewed_columns(cleaned_data[numeric_columns])
    
    for column, skewness in skewed_cols:
        if abs(skewness) > 1.0:
            cleaned_data[f'{column}_log'] = log_transform(cleaned_data, column)
            new_skewness = stats.skew(cleaned_data[f'{column}_log'].dropna())
            cleaning_report[column]['original_skewness'] = skewness
            cleaning_report[column]['transformed_skewness'] = new_skewness
    
    return cleaned_data, cleaning_report