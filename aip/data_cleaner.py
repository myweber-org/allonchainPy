
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            columns = self.numeric_columns
        
        clean_df = self.df.copy()
        for col in columns:
            if col not in self.numeric_columns:
                continue
                
            Q1 = clean_df[col].quantile(0.25)
            Q3 = clean_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            mask = (clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)
            clean_df = clean_df[mask]
        
        return clean_df
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.numeric_columns
        
        clean_df = self.df.copy()
        for col in columns:
            if col not in self.numeric_columns:
                continue
                
            z_scores = np.abs(stats.zscore(clean_df[col].dropna()))
            mask = z_scores < threshold
            clean_df = clean_df[mask]
        
        return clean_df
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        normalized_df = self.df.copy()
        for col in columns:
            if col not in self.numeric_columns:
                continue
                
            col_min = normalized_df[col].min()
            col_max = normalized_df[col].max()
            
            if col_max != col_min:
                normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
        
        return normalized_df
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        normalized_df = self.df.copy()
        for col in columns:
            if col not in self.numeric_columns:
                continue
                
            col_mean = normalized_df[col].mean()
            col_std = normalized_df[col].std()
            
            if col_std > 0:
                normalized_df[col] = (normalized_df[col] - col_mean) / col_std
        
        return normalized_df
    
    def fill_missing(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        filled_df = self.df.copy()
        for col in columns:
            if col not in self.numeric_columns:
                continue
                
            if strategy == 'mean':
                fill_value = filled_df[col].mean()
            elif strategy == 'median':
                fill_value = filled_df[col].median()
            elif strategy == 'mode':
                fill_value = filled_df[col].mode()[0]
            elif strategy == 'zero':
                fill_value = 0
            else:
                fill_value = filled_df[col].mean()
            
            filled_df[col] = filled_df[col].fillna(fill_value)
        
        return filled_df
    
    def get_summary(self):
        summary = {
            'original_shape': self.df.shape,
            'numeric_columns': self.numeric_columns,
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict()
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 200, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    data['feature_a'][np.random.choice(1000, 50)] = np.nan
    data['feature_b'][np.random.choice(1000, 30)] = np.nan
    
    outliers = np.random.choice(1000, 20)
    data['feature_a'][outliers] = data['feature_a'][outliers] * 5
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = create_sample_data()
    cleaner = DataCleaner(df)
    
    print("Data Summary:")
    summary = cleaner.get_summary()
    print(f"Original shape: {summary['original_shape']}")
    print(f"Numeric columns: {summary['numeric_columns']}")
    
    print("\nCleaning with IQR method...")
    clean_iqr = cleaner.remove_outliers_iqr()
    print(f"Cleaned shape: {clean_iqr.shape}")
    
    print("\nNormalizing data...")
    normalized = cleaner.normalize_minmax()
    print("Normalization complete")
    
    print("\nFilling missing values...")
    filled = cleaner.fill_missing(strategy='median')
    print("Missing values filled")