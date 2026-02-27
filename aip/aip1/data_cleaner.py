import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns
        self.categorical_columns = df.select_dtypes(exclude=[np.number]).columns

    def handle_missing_values(self, strategy='mean', fill_value=None):
        if strategy == 'mean' and self.numeric_columns.any():
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(
                self.df[self.numeric_columns].mean()
            )
        elif strategy == 'median' and self.numeric_columns.any():
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(
                self.df[self.numeric_columns].median()
            )
        elif strategy == 'mode' and self.categorical_columns.any():
            for col in self.categorical_columns:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        elif strategy == 'constant' and fill_value is not None:
            self.df = self.df.fillna(fill_value)
        return self

    def detect_outliers(self, method='zscore', threshold=3):
        outlier_mask = pd.Series(False, index=self.df.index)
        
        if method == 'zscore':
            for col in self.numeric_columns:
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                outlier_mask |= self.df[col].index.isin(
                    self.df[col].dropna().index[z_scores > threshold]
                )
        elif method == 'iqr':
            for col in self.numeric_columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask |= (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
        
        return outlier_mask

    def remove_outliers(self, method='zscore', threshold=3):
        outlier_mask = self.detect_outliers(method, threshold)
        self.df = self.df[~outlier_mask].reset_index(drop=True)
        return self

    def get_cleaned_data(self):
        return self.df.copy()

def example_usage():
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 8],
        'C': ['x', 'y', np.nan, 'z', 'x']
    }
    df = pd.DataFrame(sample_data)
    
    cleaner = DataCleaner(df)
    cleaned_df = (cleaner
                 .handle_missing_values(strategy='mean')
                 .remove_outliers(method='zscore', threshold=2)
                 .get_cleaned_data())
    
    return cleaned_df

if __name__ == "__main__":
    result = example_usage()
    print("Cleaned DataFrame:")
    print(result)
import numpy as np

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data