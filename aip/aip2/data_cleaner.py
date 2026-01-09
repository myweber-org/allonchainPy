import pandas as pd
import numpy as np

def remove_duplicates(df):
    """Remove duplicate rows from DataFrame."""
    return df.drop_duplicates()

def fill_missing_values(df, strategy='mean'):
    """Fill missing values using specified strategy."""
    if strategy == 'mean':
        return df.fillna(df.mean(numeric_only=True))
    elif strategy == 'median':
        return df.fillna(df.median(numeric_only=True))
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    else:
        return df.fillna(0)

def normalize_column(df, column_name):
    """Normalize specified column to range [0,1]."""
    if column_name in df.columns:
        col_min = df[column_name].min()
        col_max = df[column_name].max()
        if col_max != col_min:
            df[column_name] = (df[column_name] - col_min) / (col_max - col_min)
    return df

def clean_dataset(file_path, output_path=None):
    """Main function to clean dataset with default operations."""
    try:
        df = pd.read_csv(file_path)
        df = remove_duplicates(df)
        df = fill_missing_values(df, strategy='mean')
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df = normalize_column(df, col)
        
        if output_path:
            df.to_csv(output_path, index=False)
            return f"Cleaned data saved to {output_path}"
        return df
    except Exception as e:
        return f"Error cleaning dataset: {str(e)}"
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns
        
    def remove_outliers_zscore(self, threshold=3):
        z_scores = np.abs(stats.zscore(self.df[self.numeric_columns]))
        filtered_entries = (z_scores < threshold).all(axis=1)
        self.df = self.df[filtered_entries]
        return self
    
    def normalize_minmax(self):
        for col in self.numeric_columns:
            min_val = self.df[col].min()
            max_val = self.df[col].max()
            if max_val > min_val:
                self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        return self
    
    def fill_missing_median(self):
        for col in self.numeric_columns:
            self.df[col].fillna(self.df[col].median(), inplace=True)
        return self
    
    def get_cleaned_data(self):
        return self.df.copy()

def process_dataset(filepath):
    raw_data = pd.read_csv(filepath)
    cleaner = DataCleaner(raw_data)
    cleaned = (cleaner.fill_missing_median()
                      .remove_outliers_zscore()
                      .normalize_minmax()
                      .get_cleaned_data())
    return cleaned