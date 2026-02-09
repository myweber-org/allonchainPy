
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

def calculate_summary_stats(df, column):
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max()
    }
    return statsimport pandas as pd
import numpy as np
from typing import Optional, Dict, List

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def handle_missing_values(self, strategy: str = 'mean', columns: Optional[List[str]] = None) -> 'DataCleaner':
        if columns is None:
            columns = self.df.columns
            
        for col in columns:
            if col in self.df.columns and self.df[col].isnull().any():
                if strategy == 'mean' and pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif strategy == 'median' and pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                elif strategy == 'mode':
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                elif strategy == 'drop':
                    self.df.dropna(subset=[col], inplace=True)
                else:
                    self.df[col].fillna(0, inplace=True)
        return self
    
    def convert_types(self, type_map: Dict[str, str]) -> 'DataCleaner':
        for col, dtype in type_map.items():
            if col in self.df.columns:
                try:
                    if dtype == 'datetime':
                        self.df[col] = pd.to_datetime(self.df[col])
                    elif dtype == 'numeric':
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    elif dtype == 'category':
                        self.df[col] = self.df[col].astype('category')
                    else:
                        self.df[col] = self.df[col].astype(dtype)
                except Exception as e:
                    print(f"Error converting column {col} to {dtype}: {e}")
        return self
    
    def remove_duplicates(self, subset: Optional[List[str]] = None, keep: str = 'first') -> 'DataCleaner':
        self.df.drop_duplicates(subset=subset, keep=keep, inplace=True)
        return self
    
    def normalize_numeric(self, columns: Optional[List[str]] = None) -> 'DataCleaner':
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                col_min = self.df[col].min()
                col_max = self.df[col].max()
                if col_max > col_min:
                    self.df[col] = (self.df[col] - col_min) / (col_max - col_min)
        return self
    
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df
    
    def get_stats(self) -> Dict:
        cleaned_shape = self.df.shape
        return {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': cleaned_shape[0],
            'cleaned_columns': cleaned_shape[1],
            'rows_removed': self.original_shape[0] - cleaned_shape[0],
            'missing_values_remaining': self.df.isnull().sum().sum()
        }

def clean_csv_file(input_path: str, output_path: str, **kwargs) -> Dict:
    try:
        df = pd.read_csv(input_path)
        cleaner = DataCleaner(df)
        
        if 'missing_strategy' in kwargs:
            cleaner.handle_missing_values(strategy=kwargs['missing_strategy'])
        
        if 'type_map' in kwargs:
            cleaner.convert_types(kwargs['type_map'])
        
        if 'remove_dups' in kwargs and kwargs['remove_dups']:
            cleaner.remove_duplicates()
        
        if 'normalize' in kwargs and kwargs['normalize']:
            cleaner.normalize_numeric()
        
        cleaned_df = cleaner.get_cleaned_data()
        cleaned_df.to_csv(output_path, index=False)
        
        return {
            'success': True,
            'stats': cleaner.get_stats(),
            'output_file': output_path
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }