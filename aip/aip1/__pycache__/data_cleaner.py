import numpy as np
import pandas as pd

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape

    def remove_outliers_iqr(self, column, multiplier=1.5):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        mask = (self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)
        self.df = self.df[mask]
        removed_count = self.original_shape[0] - self.df.shape[0]
        print(f"Removed {removed_count} outliers from column '{column}'")
        return self

    def normalize_column(self, column, method='minmax'):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            if max_val != min_val:
                self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
            else:
                self.df[column] = 0
        elif method == 'zscore':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            if std_val > 0:
                self.df[column] = (self.df[column] - mean_val) / std_val
            else:
                self.df[column] = 0
        else:
            raise ValueError("Method must be 'minmax' or 'zscore'")
        
        print(f"Normalized column '{column}' using {method} method")
        return self

    def fill_missing(self, column, strategy='mean'):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        missing_count = self.df[column].isnull().sum()
        if missing_count == 0:
            print(f"No missing values in column '{column}'")
            return self
        
        if strategy == 'mean':
            fill_value = self.df[column].mean()
        elif strategy == 'median':
            fill_value = self.df[column].median()
        elif strategy == 'mode':
            fill_value = self.df[column].mode()[0]
        else:
            raise ValueError("Strategy must be 'mean', 'median', or 'mode'")
        
        self.df[column].fillna(fill_value, inplace=True)
        print(f"Filled {missing_count} missing values in column '{column}' using {strategy}")
        return self

    def get_cleaned_data(self):
        return self.df

def create_sample_data():
    np.random.seed(42)
    data = {
        'age': np.random.normal(35, 10, 100),
        'income': np.random.normal(50000, 15000, 100),
        'score': np.random.uniform(0, 100, 100)
    }
    
    df = pd.DataFrame(data)
    df.loc[10:15, 'age'] = np.nan
    df.loc[5, 'income'] = 200000
    df.loc[6, 'income'] = -50000
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    print("Original data shape:", sample_df.shape)
    
    cleaner = DataCleaner(sample_df)
    cleaned_df = (cleaner
                 .remove_outliers_iqr('income')
                 .fill_missing('age', strategy='median')
                 .normalize_column('score', method='minmax')
                 .get_cleaned_data())
    
    print("Cleaned data shape:", cleaned_df.shape)
    print("\nSummary statistics:")
    print(cleaned_df.describe())