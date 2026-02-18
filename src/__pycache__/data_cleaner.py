
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
                clean_data = clean_data[(clean_data[col] >= lower_bound) & (clean_data[col] <= upper_bound)]
        
        removed_count = self.original_shape[0] - clean_data.shape[0]
        self.data = clean_data
        return removed_count
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        normalized_data = self.data.copy()
        for col in columns:
            if col in normalized_data.columns and pd.api.types.is_numeric_dtype(normalized_data[col]):
                min_val = normalized_data[col].min()
                max_val = normalized_data[col].max()
                if max_val > min_val:
                    normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
        
        self.data = normalized_data
        return self.data
    
    def standardize_zscore(self, columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        standardized_data = self.data.copy()
        for col in columns:
            if col in standardized_data.columns and pd.api.types.is_numeric_dtype(standardized_data[col]):
                mean_val = standardized_data[col].mean()
                std_val = standardized_data[col].std()
                if std_val > 0:
                    standardized_data[col] = (standardized_data[col] - mean_val) / std_val
        
        self.data = standardized_data
        return self.data
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        filled_data = self.data.copy()
        for col in columns:
            if col in filled_data.columns and pd.api.types.is_numeric_dtype(filled_data[col]):
                median_val = filled_data[col].median()
                filled_data[col] = filled_data[col].fillna(median_val)
        
        self.data = filled_data
        return self.data
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'current_rows': self.data.shape[0],
            'columns': list(self.data.columns),
            'missing_values': self.data.isnull().sum().to_dict(),
            'numeric_columns': list(self.data.select_dtypes(include=[np.number]).columns)
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'temperature': np.random.normal(25, 5, 100),
        'humidity': np.random.uniform(30, 90, 100),
        'pressure': np.random.normal(1013, 10, 100),
        'wind_speed': np.random.exponential(5, 100)
    })
    
    data.loc[10:15, 'temperature'] = np.nan
    data.loc[95, 'humidity'] = 150
    data.loc[96, 'pressure'] = 500
    
    return data

if __name__ == "__main__":
    sample_data = create_sample_data()
    cleaner = DataCleaner(sample_data)
    
    print("Initial data shape:", cleaner.data.shape)
    print("\nMissing values before cleaning:")
    print(cleaner.data.isnull().sum())
    
    removed = cleaner.remove_outliers_iqr(['temperature', 'humidity', 'pressure', 'wind_speed'])
    print(f"\nRemoved {removed} outliers using IQR method")
    
    cleaner.fill_missing_median()
    print("\nMissing values after filling:")
    print(cleaner.data.isnull().sum())
    
    cleaner.normalize_minmax()
    print("\nData normalized using min-max scaling")
    
    summary = cleaner.get_summary()
    print("\nCleaning summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")