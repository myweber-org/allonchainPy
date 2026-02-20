
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_columns = df.columns.tolist()
        
    def remove_outliers_iqr(self, column, multiplier=1.5):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
        return self
        
    def remove_outliers_zscore(self, column, threshold=3):
        z_scores = np.abs(stats.zscore(self.df[column]))
        self.df = self.df[z_scores < threshold]
        return self
        
    def normalize_minmax(self, column):
        min_val = self.df[column].min()
        max_val = self.df[column].max()
        self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
        return self
        
    def normalize_zscore(self, column):
        mean_val = self.df[column].mean()
        std_val = self.df[column].std()
        self.df[column] = (self.df[column] - mean_val) / std_val
        return self
        
    def fill_missing_mean(self, column):
        self.df[column].fillna(self.df[column].mean(), inplace=True)
        return self
        
    def fill_missing_median(self, column):
        self.df[column].fillna(self.df[column].median(), inplace=True)
        return self
        
    def drop_duplicates(self):
        self.df.drop_duplicates(inplace=True)
        return self
        
    def get_cleaned_data(self):
        return self.df.copy()
        
    def summary(self):
        print(f"Original shape: {self.df.shape}")
        print(f"Columns: {self.original_columns}")
        print(f"Missing values:")
        print(self.df.isnull().sum())
        print(f"Data types:")
        print(self.df.dtypes)

def clean_dataset(df, config):
    cleaner = DataCleaner(df)
    
    if 'outlier_method' in config:
        method = config['outlier_method']
        columns = config.get('outlier_columns', df.select_dtypes(include=[np.number]).columns)
        
        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                if method == 'iqr':
                    cleaner.remove_outliers_iqr(col, config.get('iqr_multiplier', 1.5))
                elif method == 'zscore':
                    cleaner.remove_outliers_zscore(col, config.get('zscore_threshold', 3))
    
    if 'normalize' in config:
        method = config['normalize']
        columns = config.get('normalize_columns', df.select_dtypes(include=[np.number]).columns)
        
        for col in columns:
            if col in cleaner.df.columns and pd.api.types.is_numeric_dtype(cleaner.df[col]):
                if method == 'minmax':
                    cleaner.normalize_minmax(col)
                elif method == 'zscore':
                    cleaner.normalize_zscore(col)
    
    if 'fill_missing' in config:
        method = config['fill_missing']
        columns = config.get('missing_columns', df.columns)
        
        for col in columns:
            if col in cleaner.df.columns and cleaner.df[col].isnull().any():
                if method == 'mean':
                    cleaner.fill_missing_mean(col)
                elif method == 'median':
                    cleaner.fill_missing_median(col)
    
    if config.get('drop_duplicates', False):
        cleaner.drop_duplicates()
    
    return cleaner.get_cleaned_data()
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val == min_val:
        return data[column]
    return (data[column] - min_val) / (max_val - min_val)

def standardize_zscore(data, column):
    mean_val = data[column].mean()
    std_val = data[column].std()
    if std_val == 0:
        return data[column]
    return (data[column] - mean_val) / std_val

def clean_dataset(df, numeric_columns, outlier_removal=True, normalization='standard'):
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col not in cleaned_df.columns:
            continue
            
        if outlier_removal:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        
        if normalization == 'minmax':
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
        elif normalization == 'standard':
            cleaned_df[col] = standardize_zscore(cleaned_df, col)
    
    return cleaned_df.reset_index(drop=True)

def get_summary_statistics(df, numeric_columns):
    stats = {}
    for col in numeric_columns:
        if col in df.columns:
            stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median()
            }
    return pd.DataFrame(stats).T

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    cleaned = clean_dataset(
        sample_data, 
        ['feature1', 'feature2'], 
        outlier_removal=True, 
        normalization='standard'
    )
    
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned.shape}")
    print("\nSummary statistics:")
    print(get_summary_statistics(cleaned, ['feature1', 'feature2']))