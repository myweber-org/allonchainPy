
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
        
    def handle_missing_values(self, 
                             strategy: str = 'mean',
                             columns: Optional[List[str]] = None) -> 'DataCleaner':
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns:
                if strategy == 'mean':
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif strategy == 'median':
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                elif strategy == 'mode':
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                elif strategy == 'drop':
                    self.df = self.df.dropna(subset=[col])
                    
        return self
        
    def convert_types(self, type_map: Dict[str, type]) -> 'DataCleaner':
        for col, target_type in type_map.items():
            if col in self.df.columns:
                try:
                    self.df[col] = self.df[col].astype(target_type)
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert column {col} to {target_type}")
        return self
        
    def remove_outliers(self, 
                       columns: List[str],
                       method: str = 'iqr',
                       threshold: float = 1.5) -> 'DataCleaner':
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                if method == 'iqr':
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    self.df = self.df[(self.df[col] >= lower_bound) & 
                                     (self.df[col] <= upper_bound)]
                    
        return self
        
    def normalize_columns(self, 
                         columns: List[str],
                         method: str = 'minmax') -> 'DataCleaner':
        for col in columns:
            if col in self.df.columns:
                if method == 'minmax':
                    min_val = self.df[col].min()
                    max_val = self.df[col].max()
                    if max_val != min_val:
                        self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
                elif method == 'zscore':
                    mean_val = self.df[col].mean()
                    std_val = self.df[col].std()
                    if std_val != 0:
                        self.df[col] = (self.df[col] - mean_val) / std_val
                        
        return self
        
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df.copy()
        
    def get_cleaning_report(self) -> Dict:
        return {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': self.df.shape[0],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'columns_removed': self.original_shape[1] - self.df.shape[1]
        }

def clean_csv_file(input_path: str, 
                  output_path: str,
                  cleaning_steps: Optional[Dict] = None) -> Dict:
    try:
        df = pd.read_csv(input_path)
        cleaner = DataCleaner(df)
        
        if cleaning_steps:
            if cleaning_steps.get('remove_duplicates'):
                cleaner.remove_duplicates(cleaning_steps.get('duplicate_subset'))
                
            if cleaning_steps.get('handle_missing'):
                cleaner.handle_missing_values(
                    strategy=cleaning_steps.get('missing_strategy', 'mean'),
                    columns=cleaning_steps.get('missing_columns')
                )
                
            if cleaning_steps.get('convert_types'):
                cleaner.convert_types(cleaning_steps['convert_types'])
                
            if cleaning_steps.get('remove_outliers'):
                cleaner.remove_outliers(
                    columns=cleaning_steps.get('outlier_columns', []),
                    method=cleaning_steps.get('outlier_method', 'iqr'),
                    threshold=cleaning_steps.get('outlier_threshold', 1.5)
                )
                
            if cleaning_steps.get('normalize'):
                cleaner.normalize_columns(
                    columns=cleaning_steps.get('normalize_columns', []),
                    method=cleaning_steps.get('normalize_method', 'minmax')
                )
        
        cleaned_df = cleaner.get_cleaned_data()
        cleaned_df.to_csv(output_path, index=False)
        
        report = cleaner.get_cleaning_report()
        report['status'] = 'success'
        report['output_file'] = output_path
        
        return report
        
    except Exception as e:
        return {
            'status': 'error',
            'error_message': str(e)
        }