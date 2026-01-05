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
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        self.df = df_clean
        removed_count = self.original_shape[0] - self.df.shape[0]
        print(f"Removed {removed_count} outliers using IQR method")
        return self
        
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        
        print(f"Normalized {len(columns)} columns using Min-Max scaling")
        return self
        
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                median_val = self.df[col].median()
                missing_count = self.df[col].isna().sum()
                self.df[col].fillna(median_val, inplace=True)
                if missing_count > 0:
                    print(f"Filled {missing_count} missing values in column '{col}' with median {median_val:.2f}")
        
        return self
        
    def get_cleaned_data(self):
        return self.df
        
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'cleaned_rows': self.df.shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'numeric_columns': len(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(self.df.select_dtypes(include=['object']).columns)
        }
        return summary

def example_usage():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    data['feature_a'][:50] = np.random.normal(300, 50, 50)
    data['feature_b'][:30] = np.random.normal(500, 100, 30)
    
    df = pd.DataFrame(data)
    df.iloc[10:20, 0] = np.nan
    df.iloc[30:40, 1] = np.nan
    
    print("Original data shape:", df.shape)
    print("Missing values:", df.isna().sum().sum())
    
    cleaner = DataCleaner(df)
    cleaned_df = (cleaner
                 .remove_outliers_iqr()
                 .fill_missing_median()
                 .normalize_minmax()
                 .get_cleaned_data())
    
    summary = cleaner.get_summary()
    print("\nCleaning Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    return cleaned_df

if __name__ == "__main__":
    result_df = example_usage()
    print("\nFirst 5 rows of cleaned data:")
    print(result_df.head())import pandas as pd

def clean_dataset(df, remove_duplicates=True, fill_missing=False, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        remove_duplicates (bool): Whether to remove duplicate rows
        fill_missing (bool): Whether to fill missing values
        fill_value: Value to use for filling missing data
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if remove_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing:
        missing_before = cleaned_df.isnull().sum().sum()
        cleaned_df = cleaned_df.fillna(fill_value)
        missing_after = cleaned_df.isnull().sum().sum()
        print(f"Filled {missing_before - missing_after} missing values")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
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

def sample_data_cleaning():
    """Example usage of the data cleaning functions."""
    data = {
        'id': [1, 2, 2, 3, 4, 4],
        'value': [10, 20, 20, None, 40, 40],
        'category': ['A', 'B', 'B', 'C', 'D', 'D']
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\nDataFrame info:")
    print(df.info())
    
    is_valid, message = validate_dataframe(df, required_columns=['id', 'value'])
    print(f"\nValidation: {message}")
    
    if is_valid:
        cleaned = clean_dataset(df, remove_duplicates=True, fill_missing=True, fill_value=0)
        print("\nCleaned DataFrame:")
        print(cleaned)
        return cleaned
    
    return None

if __name__ == "__main__":
    sample_data_cleaning()import pandas as pd
import numpy as np

def clean_dataframe(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    missing_strategy (str): Strategy for handling missing values.
        Options: 'mean', 'median', 'mode', 'drop'.
    outlier_threshold (float): Number of standard deviations to identify outliers.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif missing_strategy in ['mean', 'median', 'mode']:
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            if missing_strategy == 'mean':
                fill_value = cleaned_df[column].mean()
            elif missing_strategy == 'median':
                fill_value = cleaned_df[column].median()
            else:  # mode
                fill_value = cleaned_df[column].mode()[0] if not cleaned_df[column].mode().empty else 0
            cleaned_df[column] = cleaned_df[column].fillna(fill_value)
    
    # Remove outliers for numerical columns
    numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        mean = cleaned_df[column].mean()
        std = cleaned_df[column].std()
        lower_bound = mean - outlier_threshold * std
        upper_bound = mean + outlier_threshold * std
        cleaned_df = cleaned_df[(cleaned_df[column] >= lower_bound) & 
                                (cleaned_df[column] <= upper_bound)]
    
    return cleaned_df.reset_index(drop=True)

def normalize_data(df, method='minmax'):
    """
    Normalize numerical columns in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    method (str): Normalization method. Options: 'minmax', 'zscore'.
    
    Returns:
    pd.DataFrame: Normalized DataFrame.
    """
    normalized_df = df.copy()
    numeric_columns = normalized_df.select_dtypes(include=[np.number]).columns
    
    if method == 'minmax':
        for column in numeric_columns:
            min_val = normalized_df[column].min()
            max_val = normalized_df[column].max()
            if max_val != min_val:
                normalized_df[column] = (normalized_df[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        for column in numeric_columns:
            mean = normalized_df[column].mean()
            std = normalized_df[column].std()
            if std != 0:
                normalized_df[column] = (normalized_df[column] - mean) / std
    
    return normalized_df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 9],
        'C': [10, 11, 12, 13, 14]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataframe(df, missing_strategy='mean', outlier_threshold=2)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    normalized = normalize_data(cleaned, method='minmax')
    print("\nNormalized DataFrame:")
    print(normalized)