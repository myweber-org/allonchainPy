
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self):
        self.df = self.df.drop_duplicates()
        return self
        
    def handle_missing_values(self, strategy='mean', columns=None):
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
        
    def detect_outliers_zscore(self, threshold=3, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        outliers = {}
        for col in columns:
            if col in self.df.columns:
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                outlier_indices = np.where(z_scores > threshold)[0]
                outliers[col] = outlier_indices.tolist()
                
        return outliers
        
    def remove_outliers_iqr(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
                
        return self
        
    def normalize_data(self, method='minmax', columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
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
        
    def get_cleaned_data(self):
        return self.df
        
    def get_cleaning_report(self):
        cleaned_shape = self.df.shape
        report = {
            'original_rows': self.original_shape[0],
            'cleaned_rows': cleaned_shape[0],
            'rows_removed': self.original_shape[0] - cleaned_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_columns': cleaned_shape[1]
        }
        return report

def clean_dataset(df, remove_duplicates=True, handle_missing=True, 
                  remove_outliers=True, normalize=True):
    cleaner = DataCleaner(df)
    
    if remove_duplicates:
        cleaner.remove_duplicates()
        
    if handle_missing:
        cleaner.handle_missing_values(strategy='mean')
        
    if remove_outliers:
        cleaner.remove_outliers_iqr()
        
    if normalize:
        cleaner.normalize_data(method='minmax')
        
    return cleaner.get_cleaned_data(), cleaner.get_cleaning_report()