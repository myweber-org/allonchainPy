
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
            if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        self.df = df_clean
        return self
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_norm = self.df.copy()
        for col in columns:
            if col in df_norm.columns and pd.api.types.is_numeric_dtype(df_norm[col]):
                min_val = df_norm[col].min()
                max_val = df_norm[col].max()
                if max_val > min_val:
                    df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        
        self.df = df_norm
        return self
    
    def standardize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_std = self.df.copy()
        for col in columns:
            if col in df_std.columns and pd.api.types.is_numeric_dtype(df_std[col]):
                mean_val = df_std[col].mean()
                std_val = df_std[col].std()
                if std_val > 0:
                    df_std[col] = (df_std[col] - mean_val) / std_val
        
        self.df = df_std
        return self
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_filled = self.df.copy()
        for col in columns:
            if col in df_filled.columns and pd.api.types.is_numeric_dtype(df_filled[col]):
                median_val = df_filled[col].median()
                df_filled[col] = df_filled[col].fillna(median_val)
        
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
            'removed_rows': self.get_removed_count(),
            'columns': self.df.shape[1],
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(exclude=[np.number]).columns)
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
    
    data['feature_a'][np.random.choice(1000, 20)] = np.nan
    data['feature_b'][np.random.choice(1000, 5)] = 1000
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = create_sample_data()
    print("Original data shape:", df.shape)
    
    cleaner = DataCleaner(df)
    cleaner.fill_missing_median().remove_outliers_iqr().standardize_zscore()
    
    cleaned_df = cleaner.get_cleaned_data()
    print("Cleaned data shape:", cleaned_df.shape)
    print("Rows removed:", cleaner.get_removed_count())
    
    summary = cleaner.get_summary()
    print("\nData Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
def remove_duplicates(input_list):
    """
    Remove duplicate elements from a list while preserving order.
    
    Args:
        input_list: A list containing elements (must be hashable).
    
    Returns:
        A new list with duplicates removed.
    
    Raises:
        TypeError: If elements are not hashable.
    """
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_data_with_key(input_list, key_func=None):
    """
    Remove duplicates based on a key function.
    
    Args:
        input_list: A list containing elements.
        key_func: A function that returns a hashable key for each element.
                  If None, uses the element itself.
    
    Returns:
        A new list with duplicates removed based on the key.
    """
    seen = set()
    result = []
    for item in input_list:
        key = key_func(item) if key_func else item
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, 4, 4, 5, 1, 6]
    cleaned = remove_duplicates(sample_data)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {cleaned}")
    
    sample_objects = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
        {"id": 1, "name": "Alice"},
        {"id": 3, "name": "Charlie"}
    ]
    cleaned_objects = clean_data_with_key(sample_objects, key_func=lambda x: x["id"])
    print(f"Original objects: {sample_objects}")
    print(f"Cleaned objects: {cleaned_objects}")