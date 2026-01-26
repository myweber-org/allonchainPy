
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, column, threshold=1.5):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)
        self.df = self.df[mask]
        removed_count = self.original_shape[0] - self.df.shape[0]
        return removed_count
    
    def normalize_column(self, column, method='minmax'):
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            if max_val - min_val != 0:
                self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            if std_val != 0:
                self.df[column] = (self.df[column] - mean_val) / std_val
        return self.df[column]
    
    def fill_missing_with_strategy(self, column, strategy='mean'):
        if strategy == 'mean':
            fill_value = self.df[column].mean()
        elif strategy == 'median':
            fill_value = self.df[column].median()
        elif strategy == 'mode':
            fill_value = self.df[column].mode()[0]
        else:
            fill_value = 0
        self.df[column].fillna(fill_value, inplace=True)
        return fill_value
    
    def get_cleaned_data(self):
        return self.df
    
    def get_cleaning_report(self):
        report = {
            'original_rows': self.original_shape[0],
            'cleaned_rows': self.df.shape[0],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'columns': list(self.df.columns)
        }
        return report

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.randint(1, 100, 1000)
    }
    df = pd.DataFrame(data)
    df.iloc[10:20, 0] = np.nan
    df.iloc[50:60, 1] = np.nan
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Original shape:", cleaner.original_shape)
    
    removed = cleaner.remove_outliers_iqr('feature_a')
    print(f"Removed {removed} outliers from feature_a")
    
    cleaner.normalize_column('feature_b', method='minmax')
    cleaner.fill_missing_with_strategy('feature_a', strategy='mean')
    cleaner.fill_missing_with_strategy('feature_b', strategy='median')
    
    cleaned_df = cleaner.get_cleaned_data()
    report = cleaner.get_cleaning_report()
    
    print("Cleaned shape:", cleaned_df.shape)
    print("Cleaning report:", report)
    print("First 5 rows of cleaned data:")
    print(cleaned_df.head())
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path):
    data = pd.read_csv(file_path)
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        data = remove_outliers_iqr(data, col)
        data = normalize_minmax(data, col)
    
    return data

if __name__ == "__main__":
    cleaned_data = clean_dataset('raw_data.csv')
    cleaned_data.to_csv('cleaned_data.csv', index=False)
    print(f"Data cleaning complete. Original shape: {pd.read_csv('raw_data.csv').shape}, Cleaned shape: {cleaned_data.shape}")