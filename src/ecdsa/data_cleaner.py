
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_missing(self, threshold=0.3):
        missing_percent = self.df.isnull().sum() / len(self.df)
        columns_to_drop = missing_percent[missing_percent > threshold].index
        self.df = self.df.drop(columns=columns_to_drop)
        return self
    
    def fill_numeric_missing(self, method='median'):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.df[col].isnull().any():
                if method == 'median':
                    fill_value = self.df[col].median()
                elif method == 'mean':
                    fill_value = self.df[col].mean()
                elif method == 'mode':
                    fill_value = self.df[col].mode()[0]
                else:
                    fill_value = 0
                
                self.df[col] = self.df[col].fillna(fill_value)
        
        return self
    
    def detect_outliers_zscore(self, threshold=3):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outliers_mask = pd.Series([False] * len(self.df))
        
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(self.df[col].dropna()))
            col_outliers = z_scores > threshold
            outliers_mask = outliers_mask | col_outliers.reindex(self.df.index, fill_value=False)
        
        return outliers_mask
    
    def remove_outliers(self, threshold=3):
        outliers_mask = self.detect_outliers_zscore(threshold)
        self.df = self.df[~outliers_mask]
        return self
    
    def normalize_data(self, method='minmax'):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if method == 'minmax':
            for col in numeric_cols:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        
        elif method == 'standard':
            for col in numeric_cols:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
        
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_summary(self):
        cleaned_shape = self.df.shape
        rows_removed = self.original_shape[0] - cleaned_shape[0]
        cols_removed = self.original_shape[1] - cleaned_shape[1]
        
        summary = {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': cleaned_shape[0],
            'cleaned_columns': cleaned_shape[1],
            'rows_removed': rows_removed,
            'columns_removed': cols_removed,
            'missing_values': self.df.isnull().sum().sum(),
            'numeric_columns': len(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(self.df.select_dtypes(include=['object']).columns)
        }
        
        return summary

def clean_dataset(df, outlier_threshold=3, normalize=True):
    cleaner = DataCleaner(df)
    
    cleaner.remove_missing(threshold=0.3)
    cleaner.fill_numeric_missing(method='median')
    cleaner.remove_outliers(threshold=outlier_threshold)
    
    if normalize:
        cleaner.normalize_data(method='minmax')
    
    return cleaner.get_cleaned_data(), cleaner.get_summary()