import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using the Interquartile Range method.
    Returns a boolean mask for outliers.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    return (data[column] < lower_bound) | (data[column] > upper_bound)

def remove_outliers(df, columns, threshold=1.5):
    """
    Remove outliers from specified columns using IQR method.
    Returns cleaned DataFrame.
    """
    df_clean = df.copy()
    for col in columns:
        outliers = detect_outliers_iqr(df_clean, col, threshold)
        df_clean = df_clean[~outliers]
    return df_clean.reset_index(drop=True)

def normalize_minmax(data, columns):
    """
    Apply min-max normalization to specified columns.
    Returns DataFrame with normalized columns.
    """
    df_normalized = data.copy()
    for col in columns:
        min_val = df_normalized[col].min()
        max_val = df_normalized[col].max()
        if max_val > min_val:
            df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
    return df_normalized

def standardize_zscore(data, columns):
    """
    Apply z-score standardization to specified columns.
    Returns DataFrame with standardized columns.
    """
    df_standardized = data.copy()
    for col in columns:
        mean_val = df_standardized[col].mean()
        std_val = df_standardized[col].std()
        if std_val > 0:
            df_standardized[col] = (df_standardized[col] - mean_val) / std_val
    return df_standardized

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy.
    Supported strategies: 'mean', 'median', 'mode', 'drop'
    """
    df_processed = df.copy()
    
    if columns is None:
        columns = df_processed.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        return df_processed.dropna(subset=columns)
    
    for col in columns:
        if df_processed[col].isnull().any():
            if strategy == 'mean':
                fill_value = df_processed[col].mean()
            elif strategy == 'median':
                fill_value = df_processed[col].median()
            elif strategy == 'mode':
                fill_value = df_processed[col].mode()[0]
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")
            
            df_processed[col] = df_processed[col].fillna(fill_value)
    
    return df_processed

def validate_dataframe(df, required_columns=None, numeric_columns=None):
    """
    Basic DataFrame validation.
    Returns tuple of (is_valid, error_message)
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    if numeric_columns:
        non_numeric = [col for col in numeric_columns if not np.issubdtype(df[col].dtype, np.number)]
        if non_numeric:
            return False, f"Non-numeric columns specified as numeric: {non_numeric}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    return True, "DataFrame validation passed"
import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns
        self.categorical_columns = df.select_dtypes(exclude=[np.number]).columns

    def handle_missing_values(self, strategy='mean', fill_value=None):
        if strategy == 'mean':
            for col in self.numeric_columns:
                self.df[col].fillna(self.df[col].mean(), inplace=True)
        elif strategy == 'median':
            for col in self.numeric_columns:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        elif strategy == 'mode':
            for col in self.df.columns:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        elif strategy == 'constant':
            if fill_value is not None:
                self.df.fillna(fill_value, inplace=True)
            else:
                raise ValueError("fill_value must be provided for constant strategy")
        else:
            raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'mode', or 'constant'")
        return self.df

    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            columns = self.numeric_columns
        
        df_clean = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        self.df = df_clean
        return self.df

    def standardize_data(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        df_scaled = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                mean = df_scaled[col].mean()
                std = df_scaled[col].std()
                if std > 0:
                    df_scaled[col] = (df_scaled[col] - mean) / std
        self.df = df_scaled
        return self.df

    def get_cleaned_data(self):
        return self.df.copy()

def example_usage():
    data = {
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, 20, 30, np.nan, 50, 60],
        'C': ['X', 'Y', 'X', 'Y', 'X', 'Y']
    }
    df = pd.DataFrame(data)
    
    cleaner = DataCleaner(df)
    print("Original data:")
    print(df)
    
    cleaner.handle_missing_values(strategy='mean')
    cleaner.remove_outliers_iqr(multiplier=1.5)
    cleaned_df = cleaner.get_cleaned_data()
    
    print("\nCleaned data:")
    print(cleaned_df)
    
    return cleaned_df

if __name__ == "__main__":
    example_usage()