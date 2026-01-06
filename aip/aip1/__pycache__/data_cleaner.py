
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    def handle_missing_values(self, strategy='mean', fill_value=None):
        if strategy == 'mean':
            for col in self.numeric_columns:
                self.df[col].fillna(self.df[col].mean(), inplace=True)
        elif strategy == 'median':
            for col in self.numeric_columns:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        elif strategy == 'mode':
            for col in self.numeric_columns:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        elif strategy == 'constant' and fill_value is not None:
            self.df.fillna(fill_value, inplace=True)
        
        for col in self.categorical_columns:
            self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        
        return self.df
    
    def remove_outliers(self, method='zscore', threshold=3):
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(self.df[self.numeric_columns]))
            mask = (z_scores < threshold).all(axis=1)
            self.df = self.df[mask]
        elif method == 'iqr':
            for col in self.numeric_columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        
        return self.df
    
    def get_cleaned_data(self):
        return self.df
    
    def get_summary(self):
        summary = {
            'original_rows': len(self.df),
            'original_columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().sum(),
            'numeric_columns': len(self.numeric_columns),
            'categorical_columns': len(self.categorical_columns)
        }
        return summary

def clean_dataset(df, missing_strategy='mean', outlier_method=None):
    cleaner = DataCleaner(df)
    cleaner.handle_missing_values(strategy=missing_strategy)
    
    if outlier_method:
        cleaner.remove_outliers(method=outlier_method)
    
    return cleaner.get_cleaned_data()
import pandas as pd
import numpy as np

def remove_duplicates(df):
    """
    Remove duplicate rows from a DataFrame.
    """
    return df.drop_duplicates()

def fill_missing_values(df, strategy='mean'):
    """
    Fill missing values in numeric columns.
    """
    df_filled = df.copy()
    numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        for col in numeric_cols:
            df_filled[col].fillna(df_filled[col].mean(), inplace=True)
    elif strategy == 'median':
        for col in numeric_cols:
            df_filled[col].fillna(df_filled[col].median(), inplace=True)
    elif strategy == 'mode':
        for col in numeric_cols:
            df_filled[col].fillna(df_filled[col].mode()[0], inplace=True)
    
    return df_filled

def normalize_column(df, column_name):
    """
    Normalize a numeric column to range [0, 1].
    """
    if column_name in df.columns:
        col_min = df[column_name].min()
        col_max = df[column_name].max()
        
        if col_max != col_min:
            df[column_name] = (df[column_name] - col_min) / (col_max - col_min)
    
    return df

def clean_dataset(file_path, output_path=None):
    """
    Main function to clean a dataset from a CSV file.
    """
    df = pd.read_csv(file_path)
    
    df = remove_duplicates(df)
    df = fill_missing_values(df, strategy='median')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df = normalize_column(df, col)
    
    if output_path:
        df.to_csv(output_path, index=False)
    
    return df