
import pandas as pd
import numpy as np
from pathlib import Path

class DataCleaner:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.df = None
        
    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Loaded data with shape: {self.df.shape}")
            return True
        except FileNotFoundError:
            print(f"Error: File not found at {self.file_path}")
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for column in columns:
            if column in self.df.columns:
                missing_count = self.df[column].isnull().sum()
                if missing_count > 0:
                    if strategy == 'mean':
                        fill_value = self.df[column].mean()
                    elif strategy == 'median':
                        fill_value = self.df[column].median()
                    elif strategy == 'mode':
                        fill_value = self.df[column].mode()[0]
                    elif strategy == 'drop':
                        self.df = self.df.dropna(subset=[column])
                        print(f"Dropped rows with missing values in column: {column}")
                        continue
                    else:
                        fill_value = 0
                    
                    self.df[column].fillna(fill_value, inplace=True)
                    print(f"Filled {missing_count} missing values in column '{column}' with {strategy}: {fill_value}")
    
    def remove_duplicates(self, subset=None, keep='first'):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
        
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        removed = initial_rows - len(self.df)
        print(f"Removed {removed} duplicate rows")
    
    def normalize_column(self, column, method='minmax'):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
        
        if column not in self.df.columns:
            print(f"Column '{column}' not found in data")
            return
        
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            if max_val != min_val:
                self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
                print(f"Normalized column '{column}' using min-max scaling")
            else:
                print(f"Cannot normalize column '{column}' - all values are identical")
        
        elif method == 'zscore':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            if std_val > 0:
                self.df[column] = (self.df[column] - mean_val) / std_val
                print(f"Normalized column '{column}' using z-score normalization")
            else:
                print(f"Cannot normalize column '{column}' - standard deviation is zero")
    
    def save_cleaned_data(self, output_path=None):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
        
        if output_path is None:
            output_path = self.file_path.parent / f"cleaned_{self.file_path.name}"
        
        self.df.to_csv(output_path, index=False)
        print(f"Saved cleaned data to: {output_path}")
        return output_path
    
    def get_summary(self):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
        
        summary = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'numeric_stats': self.df.describe().to_dict() if len(self.df.select_dtypes(include=[np.number]).columns) > 0 else {}
        }
        return summary

def clean_csv_file(input_file, output_file=None, missing_strategy='mean'):
    cleaner = DataCleaner(input_file)
    
    if cleaner.load_data():
        cleaner.handle_missing_values(strategy=missing_strategy)
        cleaner.remove_duplicates()
        
        numeric_cols = cleaner.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            cleaner.normalize_column(col, method='minmax')
        
        saved_path = cleaner.save_cleaned_data(output_file)
        return saved_path, cleaner.get_summary()
    
    return None, None

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10.5, None, 15.2, 20.1, None, 10.5],
        'category': ['A', 'B', 'A', 'C', 'B', 'A']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_file = 'test_data.csv'
    test_df.to_csv(test_file, index=False)
    
    result_file, summary = clean_csv_file(test_file, 'cleaned_test_data.csv')
    
    if result_file:
        print(f"Data cleaning completed. Results saved to: {result_file}")
        print(f"Summary: {summary}")
    
    Path(test_file).unlink(missing_ok=True)