import pandas as pd
import re

def clean_text_column(df, column_name):
    """
    Standardize text by converting to lowercase, removing extra spaces,
    and eliminating special characters.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = df[column_name].astype(str)
    df[column_name] = df[column_name].str.lower()
    df[column_name] = df[column_name].str.strip()
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    return df

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def standardize_data(df, text_columns=None):
    """
    Apply cleaning operations to multiple text columns.
    """
    df_clean = df.copy()
    
    if text_columns:
        for col in text_columns:
            df_clean = clean_text_column(df_clean, col)
    
    df_clean = remove_duplicates(df_clean)
    return df_clean

if __name__ == "__main__":
    sample_data = {
        'name': ['John Doe', 'Jane Smith', 'JOHN DOE', 'Alice Johnson  ', 'bob@example'],
        'email': ['john@test.com', 'jane@test.com', 'JOHN@TEST.COM', 'alice@test.com', 'bob@test.com'],
        'age': [25, 30, 25, 28, 35]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = standardize_data(df, text_columns=['name', 'email'])
    print("\nCleaned DataFrame:")
    print(cleaned_df)
def filter_valid_entries(data, required_keys):
    """
    Filters a list of dictionaries, returning only those that contain all specified keys
    and where none of the required key values are None or empty strings.
    """
    if not isinstance(data, list):
        raise TypeError("Input data must be a list")
    if not isinstance(required_keys, list):
        raise TypeError("Required keys must be a list")

    valid_entries = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        try:
            if all(key in entry and entry[key] is not None and entry[key] != "" for key in required_keys):
                valid_entries.append(entry)
        except (TypeError, KeyError):
            continue
    return valid_entries
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, threshold=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_clean = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                mask = (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)
                df_clean = df_clean[mask]
        
        removed_count = self.original_shape[0] - df_clean.shape[0]
        self.df = df_clean.reset_index(drop=True)
        return removed_count
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_normalized = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    df_normalized[col] = (self.df[col] - min_val) / (max_val - min_val)
        
        self.df = df_normalized
        return self
    
    def standardize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_standardized = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    df_standardized[col] = (self.df[col] - mean_val) / std_val
        
        self.df = df_standardized
        return self
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_filled = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                median_val = self.df[col].median()
                df_filled[col] = self.df[col].fillna(median_val)
        
        self.df = df_filled
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
            'missing_values': self.df.isnull().sum().sum()
        }
        return summary

def process_dataset(file_path, output_path=None):
    try:
        df = pd.read_csv(file_path)
        cleaner = DataCleaner(df)
        
        print(f"Original dataset shape: {cleaner.original_shape}")
        
        removed = cleaner.remove_outliers_iqr()
        print(f"Removed {removed} outliers using IQR method")
        
        cleaner.fill_missing_median()
        cleaner.standardize_zscore()
        
        summary = cleaner.get_summary()
        print(f"Cleaned dataset shape: {cleaner.df.shape}")
        print(f"Missing values after cleaning: {summary['missing_values']}")
        
        if output_path:
            cleaner.df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return cleaner.get_cleaned_data()
        
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return None