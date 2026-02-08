
import pandas as pd
import numpy as np
from typing import Optional

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
    
    def remove_duplicates(self, subset: Optional[list] = None) -> 'DataCleaner':
        self.df = self.df.drop_duplicates(subset=subset, keep='first')
        return self
    
    def handle_missing_values(self, strategy: str = 'mean', fill_value: Optional[float] = None) -> 'DataCleaner':
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if strategy == 'mean':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        elif strategy == 'median':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        elif strategy == 'mode':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mode().iloc[0])
        elif strategy == 'constant' and fill_value is not None:
            self.df[numeric_cols] = self.df[numeric_cols].fillna(fill_value)
        elif strategy == 'drop':
            self.df = self.df.dropna(subset=numeric_cols)
        
        return self
    
    def normalize_numeric(self, method: str = 'minmax') -> 'DataCleaner':
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if method == 'minmax':
            for col in numeric_cols:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            for col in numeric_cols:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
        
        return self
    
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df
    
    def get_cleaning_report(self) -> dict:
        final_shape = self.df.shape
        return {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'final_rows': final_shape[0],
            'final_columns': final_shape[1],
            'rows_removed': self.original_shape[0] - final_shape[0],
            'missing_values_remaining': self.df.isnull().sum().sum()
        }

def clean_csv_file(input_path: str, output_path: str, **kwargs) -> dict:
    df = pd.read_csv(input_path)
    cleaner = DataCleaner(df)
    
    if 'remove_duplicates' in kwargs and kwargs['remove_duplicates']:
        cleaner.remove_duplicates(kwargs.get('duplicate_subset'))
    
    if 'handle_missing' in kwargs and kwargs['handle_missing']:
        cleaner.handle_missing_values(
            strategy=kwargs.get('missing_strategy', 'mean'),
            fill_value=kwargs.get('fill_value')
        )
    
    if 'normalize' in kwargs and kwargs['normalize']:
        cleaner.normalize_numeric(method=kwargs.get('normalize_method', 'minmax'))
    
    cleaned_df = cleaner.get_cleaned_data()
    cleaned_df.to_csv(output_path, index=False)
    
    return cleaner.get_cleaning_report()