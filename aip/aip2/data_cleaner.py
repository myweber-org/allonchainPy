
import pandas as pd
import numpy as np
from pathlib import Path

class DataCleaner:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.df = None
        self.load_data()
    
    def load_data(self):
        if self.file_path.exists():
            self.df = pd.read_csv(self.file_path)
            print(f"Loaded data with shape: {self.df.shape}")
        else:
            raise FileNotFoundError(f"File not found: {self.file_path}")
    
    def identify_missing(self):
        missing_summary = self.df.isnull().sum()
        missing_percentage = (missing_summary / len(self.df)) * 100
        missing_report = pd.DataFrame({
            'missing_count': missing_summary,
            'missing_percentage': missing_percentage
        })
        return missing_report[missing_report['missing_count'] > 0]
    
    def fill_missing_numeric(self, strategy='mean'):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'zero':
                    fill_value = 0
                else:
                    raise ValueError("Invalid strategy. Use 'mean', 'median', or 'zero'")
                
                self.df[col].fillna(fill_value, inplace=True)
                print(f"Filled missing values in {col} with {strategy}: {fill_value}")
    
    def fill_missing_categorical(self, strategy='mode'):
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.df[col].isnull().any():
                if strategy == 'mode':
                    fill_value = self.df[col].mode()[0]
                elif strategy == 'unknown':
                    fill_value = 'unknown'
                else:
                    raise ValueError("Invalid strategy. Use 'mode' or 'unknown'")
                
                self.df[col].fillna(fill_value, inplace=True)
                print(f"Filled missing values in {col} with {strategy}: {fill_value}")
    
    def drop_columns(self, threshold=50):
        missing_report = self.identify_missing()
        columns_to_drop = missing_report[missing_report['missing_percentage'] > threshold].index
        if len(columns_to_drop) > 0:
            self.df.drop(columns=columns_to_drop, inplace=True)
            print(f"Dropped columns with >{threshold}% missing values: {list(columns_to_drop)}")
    
    def save_cleaned_data(self, output_path=None):
        if output_path is None:
            output_path = self.file_path.parent / f"cleaned_{self.file_path.name}"
        
        self.df.to_csv(output_path, index=False)
        print(f"Saved cleaned data to: {output_path}")
        return output_path

def process_csv_file(input_file, output_dir='cleaned_data'):
    cleaner = DataCleaner(input_file)
    
    print("Initial missing values report:")
    print(cleaner.identify_missing())
    
    cleaner.drop_columns(threshold=50)
    cleaner.fill_missing_numeric(strategy='mean')
    cleaner.fill_missing_categorical(strategy='mode')
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    output_path = cleaner.save_cleaned_data(output_dir / f"cleaned_{Path(input_file).name}")
    return output_path

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5],
        'age': [25, np.nan, 30, 35, np.nan],
        'salary': [50000, 60000, np.nan, 80000, 90000],
        'department': ['IT', 'HR', 'IT', np.nan, 'Finance'],
        'experience': [2, 5, np.nan, np.nan, 10]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_file = 'test_data.csv'
    test_df.to_csv(test_file, index=False)
    
    try:
        result = process_csv_file(test_file)
        print(f"Processing completed. Output: {result}")
    finally:
        if Path(test_file).exists():
            Path(test_file).unlink()