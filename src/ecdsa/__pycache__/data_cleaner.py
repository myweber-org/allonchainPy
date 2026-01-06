import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    def remove_missing(self, threshold=0.8):
        self.df = self.df.dropna(thresh=len(self.df.columns) * threshold)
        return self
    
    def fill_numeric_missing(self, method='median'):
        for col in self.numeric_columns:
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
    
    def remove_outliers_zscore(self, threshold=3):
        for col in self.numeric_columns:
            z_scores = np.abs(stats.zscore(self.df[col]))
            self.df = self.df[z_scores < threshold]
        return self
    
    def normalize_data(self, method='minmax'):
        if method == 'minmax':
            for col in self.numeric_columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        elif method == 'standard':
            for col in self.numeric_columns:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
        return self
    
    def get_cleaned_data(self):
        return self.df.copy()

def process_dataset(file_path, output_path=None):
    try:
        df = pd.read_csv(file_path)
        cleaner = DataCleaner(df)
        cleaned_df = (cleaner
                     .remove_missing(threshold=0.7)
                     .fill_numeric_missing(method='median')
                     .remove_outliers_zscore(threshold=3)
                     .normalize_data(method='minmax')
                     .get_cleaned_data())
        
        if output_path:
            cleaned_df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to {output_path}")
        
        return cleaned_df
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return None

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 100, 7, 8, 9, 10],
        'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    })
    
    cleaner = DataCleaner(sample_data)
    result = cleaner.remove_outliers_zscore().normalize_data().get_cleaned_data()
    print("Original shape:", sample_data.shape)
    print("Cleaned shape:", result.shape)
    print("Cleaned data preview:")
    print(result.head())