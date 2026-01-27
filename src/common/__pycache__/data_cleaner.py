
import pandas as pd
import numpy as np
from typing import Optional, Dict, List

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> 'DataCleaner':
        self.df = self.df.drop_duplicates(subset=subset)
        return self
        
    def handle_missing_values(self, strategy: str = 'mean', 
                             custom_values: Optional[Dict] = None) -> 'DataCleaner':
        if strategy == 'drop':
            self.df = self.df.dropna()
        elif strategy == 'mean':
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        elif strategy == 'custom' and custom_values:
            self.df = self.df.fillna(custom_values)
        return self
        
    def convert_dtypes(self, dtype_map: Dict[str, str]) -> 'DataCleaner':
        for col, dtype in dtype_map.items():
            if col in self.df.columns:
                try:
                    if dtype == 'datetime':
                        self.df[col] = pd.to_datetime(self.df[col])
                    else:
                        self.df[col] = self.df[col].astype(dtype)
                except Exception as e:
                    print(f"Warning: Could not convert {col} to {dtype}: {e}")
        return self
        
    def remove_outliers(self, column: str, method: str = 'iqr', 
                       threshold: float = 1.5) -> 'DataCleaner':
        if column not in self.df.columns:
            return self
            
        if method == 'iqr':
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            self.df = self.df[(self.df[column] >= lower_bound) & 
                             (self.df[column] <= upper_bound)]
        return self
        
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df
        
    def get_cleaning_report(self) -> Dict:
        return {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': self.df.shape[0],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'columns_removed': self.original_shape[1] - self.df.shape[1]
        }

def clean_csv_file(input_path: str, output_path: str, 
                  cleaning_steps: Optional[Dict] = None) -> Dict:
    try:
        df = pd.read_csv(input_path)
        cleaner = DataCleaner(df)
        
        if cleaning_steps:
            if cleaning_steps.get('remove_duplicates'):
                cleaner.remove_duplicates(cleaning_steps.get('duplicate_subset'))
                
            if cleaning_steps.get('handle_missing'):
                strategy = cleaning_steps.get('missing_strategy', 'mean')
                custom_vals = cleaning_steps.get('custom_values')
                cleaner.handle_missing_values(strategy, custom_vals)
                
            if cleaning_steps.get('convert_dtypes'):
                cleaner.convert_dtypes(cleaning_steps['convert_dtypes'])
                
            if cleaning_steps.get('remove_outliers'):
                for outlier_config in cleaning_steps['remove_outliers']:
                    cleaner.remove_outliers(**outlier_config)
        
        cleaned_df = cleaner.get_cleaned_data()
        cleaned_df.to_csv(output_path, index=False)
        
        report = cleaner.get_cleaning_report()
        report['success'] = True
        return report
        
    except Exception as e:
        return {'success': False, 'error': str(e)}