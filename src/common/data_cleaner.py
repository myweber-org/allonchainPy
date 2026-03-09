
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_df = df.copy()
        
    def remove_outliers_iqr(self, column):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
        return self
        
    def remove_outliers_zscore(self, column, threshold=3):
        z_scores = np.abs(stats.zscore(self.df[column]))
        self.df = self.df[z_scores < threshold]
        return self
        
    def normalize_column(self, column, method='minmax'):
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
        elif method == 'standard':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            self.df[column] = (self.df[column] - mean_val) / std_val
        return self
        
    def fill_missing(self, column, method='mean'):
        if method == 'mean':
            fill_value = self.df[column].mean()
        elif method == 'median':
            fill_value = self.df[column].median()
        elif method == 'mode':
            fill_value = self.df[column].mode()[0]
        else:
            fill_value = method
            
        self.df[column].fillna(fill_value, inplace=True)
        return self
        
    def get_cleaned_data(self):
        return self.df
        
    def reset_to_original(self):
        self.df = self.original_df.copy()
        return self
        
    def summary(self):
        print(f"Original rows: {len(self.original_df)}")
        print(f"Cleaned rows: {len(self.df)}")
        print(f"Removed rows: {len(self.original_df) - len(self.df)}")
        print("\nMissing values:")
        print(self.df.isnull().sum())
        print("\nData types:")
        print(self.df.dtypes)

def clean_dataset(df, config):
    cleaner = DataCleaner(df)
    
    for column, operations in config.items():
        for operation, params in operations.items():
            if operation == 'remove_outliers':
                method = params.get('method', 'iqr')
                if method == 'iqr':
                    cleaner.remove_outliers_iqr(column)
                elif method == 'zscore':
                    threshold = params.get('threshold', 3)
                    cleaner.remove_outliers_zscore(column, threshold)
                    
            elif operation == 'normalize':
                method = params.get('method', 'minmax')
                cleaner.normalize_column(column, method)
                
            elif operation == 'fill_missing':
                method = params.get('method', 'mean')
                cleaner.fill_missing(column, method)
    
    return cleaner.get_cleaned_data()
import pandas as pd

def clean_dataset(df, sort_column=None):
    """
    Clean a pandas DataFrame by removing duplicate rows and optionally sorting.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        sort_column (str, optional): Column name to sort by. Defaults to None.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with duplicates removed and sorted if specified.
    """
    cleaned_df = df.drop_duplicates()
    
    if sort_column and sort_column in cleaned_df.columns:
        cleaned_df = cleaned_df.sort_values(by=sort_column)
    
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and required columns.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of required column names.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 4],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', 'David'],
        'score': [85, 92, 92, 78, 95, 95]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned = clean_dataset(df, sort_column='score')
    print(cleaned)