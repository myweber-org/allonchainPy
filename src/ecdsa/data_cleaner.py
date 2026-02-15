
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, data):
        self.data = data
        self.original_shape = data.shape
        
    def remove_outliers_iqr(self, columns=None, factor=1.5):
        if columns is None:
            columns = self.data.columns
            
        clean_data = self.data.copy()
        
        for col in columns:
            if pd.api.types.is_numeric_dtype(clean_data[col]):
                Q1 = clean_data[col].quantile(0.25)
                Q3 = clean_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                mask = (clean_data[col] >= lower_bound) & (clean_data[col] <= upper_bound)
                clean_data = clean_data[mask]
                
        return clean_data
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.data.columns
            
        clean_data = self.data.copy()
        
        for col in columns:
            if pd.api.types.is_numeric_dtype(clean_data[col]):
                z_scores = np.abs(stats.zscore(clean_data[col].dropna()))
                mask = z_scores < threshold
                clean_data = clean_data[mask]
                
        return clean_data
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        normalized_data = self.data.copy()
        
        for col in columns:
            if pd.api.types.is_numeric_dtype(normalized_data[col]):
                min_val = normalized_data[col].min()
                max_val = normalized_data[col].max()
                
                if max_val > min_val:
                    normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
                    
        return normalized_data
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        normalized_data = self.data.copy()
        
        for col in columns:
            if pd.api.types.is_numeric_dtype(normalized_data[col]):
                mean_val = normalized_data[col].mean()
                std_val = normalized_data[col].std()
                
                if std_val > 0:
                    normalized_data[col] = (normalized_data[col] - mean_val) / std_val
                    
        return normalized_data
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.data.columns
            
        filled_data = self.data.copy()
        
        for col in columns:
            if pd.api.types.is_numeric_dtype(filled_data[col]):
                if strategy == 'mean':
                    fill_value = filled_data[col].mean()
                elif strategy == 'median':
                    fill_value = filled_data[col].median()
                elif strategy == 'mode':
                    fill_value = filled_data[col].mode()[0]
                else:
                    fill_value = 0
                    
                filled_data[col] = filled_data[col].fillna(fill_value)
            else:
                filled_data[col] = filled_data[col].fillna('Unknown')
                
        return filled_data
    
    def get_cleaning_report(self):
        report = {
            'original_shape': self.original_shape,
            'current_shape': self.data.shape,
            'missing_values': self.data.isnull().sum().to_dict(),
            'numeric_columns': list(self.data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.data.select_dtypes(exclude=[np.number]).columns)
        }
        return report