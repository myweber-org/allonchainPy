
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
    
    def handle_missing_values(self, strategy: str = 'drop', fill_value: Optional[float] = None) -> 'DataCleaner':
        if strategy == 'drop':
            self.df = self.df.dropna()
        elif strategy == 'fill':
            if fill_value is not None:
                self.df = self.df.fillna(fill_value)
            else:
                self.df = self.df.fillna(self.df.mean(numeric_only=True))
        return self
    
    def remove_outliers_iqr(self, column: str, multiplier: float = 1.5) -> 'DataCleaner':
        if column not in self.df.columns:
            return self
        
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
        return self
    
    def normalize_column(self, column: str) -> 'DataCleaner':
        if column not in self.df.columns:
            return self
        
        col_min = self.df[column].min()
        col_max = self.df[column].max()
        
        if col_max != col_min:
            self.df[column] = (self.df[column] - col_min) / (col_max - col_min)
        
        return self
    
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df
    
    def get_cleaning_report(self) -> dict:
        removed_rows = self.original_shape[0] - self.df.shape[0]
        removed_cols = self.original_shape[1] - self.df.shape[1]
        
        return {
            'original_shape': self.original_shape,
            'cleaned_shape': self.df.shape,
            'rows_removed': removed_rows,
            'columns_removed': removed_cols,
            'remaining_columns': list(self.df.columns)
        }

def clean_dataset(df: pd.DataFrame, 
                  remove_dups: bool = True,
                  handle_nulls: str = 'drop',
                  outlier_columns: Optional[List[str]] = None) -> pd.DataFrame:
    
    cleaner = DataCleaner(df)
    
    if remove_dups:
        cleaner.remove_duplicates()
    
    cleaner.handle_missing_values(strategy=handle_nulls)
    
    if outlier_columns:
        for col in outlier_columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                cleaner.remove_outliers_iqr(col)
    
    return cleaner.get_cleaned_data()