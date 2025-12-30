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
            
    def handle_missing_values(self, strategy='mean', columns=None):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
            
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for column in columns:
            if column in self.df.columns:
                if self.df[column].isnull().any():
                    if strategy == 'mean':
                        fill_value = self.df[column].mean()
                    elif strategy == 'median':
                        fill_value = self.df[column].median()
                    elif strategy == 'mode':
                        fill_value = self.df[column].mode()[0]
                    else:
                        fill_value = 0
                        
                    self.df[column].fillna(fill_value, inplace=True)
                    print(f"Filled missing values in {column} using {strategy}")
                    
    def remove_duplicates(self):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
            
        initial_count = len(self.df)
        self.df.drop_duplicates(inplace=True)
        removed = initial_count - len(self.df)
        print(f"Removed {removed} duplicate rows")
        
    def normalize_column(self, column_name):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
            
        if column_name in self.df.columns:
            if self.df[column_name].dtype in [np.float64, np.int64]:
                min_val = self.df[column_name].min()
                max_val = self.df[column_name].max()
                
                if max_val > min_val:
                    self.df[column_name] = (self.df[column_name] - min_val) / (max_val - min_val)
                    print(f"Normalized column {column_name}")
                else:
                    print(f"Cannot normalize {column_name}: constant values")
            else:
                print(f"Cannot normalize non-numeric column: {column_name}")
                
    def save_cleaned_data(self, output_path=None):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
            
        if output_path is None:
            output_path = self.file_path.parent / f"cleaned_{self.file_path.name}"
            
        self.df.to_csv(output_path, index=False)
        print(f"Saved cleaned data to {output_path}")
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
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns)
        }
        return summary

def clean_dataset(input_file, output_file=None):
    cleaner = DataCleaner(input_file)
    
    if cleaner.load_data():
        cleaner.handle_missing_values(strategy='mean')
        cleaner.remove_duplicates()
        
        summary = cleaner.get_summary()
        for column in summary['numeric_columns']:
            cleaner.normalize_column(column)
            
        saved_path = cleaner.save_cleaned_data(output_file)
        return saved_path
    else:
        return None

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10.5, None, 15.2, 20.1, 8.7, 8.7],
        'category': ['A', 'B', 'A', 'C', 'B', 'B']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_file = "test_data.csv"
    test_df.to_csv(test_file, index=False)
    
    result = clean_dataset(test_file, "cleaned_test_data.csv")
    
    if result:
        print(f"Data cleaning completed. Result saved to: {result}")
        
    Path(test_file).unlink(missing_ok=True)