
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, factor=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_clean = self.df.copy()
        for col in columns:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                mask = (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)
                df_clean = df_clean[mask]
        
        self.df = df_clean.reset_index(drop=True)
        return self
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        
        return self
    
    def standardize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
        
        return self
    
    def fill_missing_mean(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].mean())
        
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'feature_c': np.random.uniform(0, 1, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    data['feature_a'][[10, 25, 50]] = [500, -200, 300]
    data['feature_b'][[15, 30]] = [1000, 1200]
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = create_sample_data()
    print(f"Original data shape: {df.shape}")
    
    cleaner = DataCleaner(df)
    cleaner.remove_outliers_iqr(['feature_a', 'feature_b'])
    cleaner.fill_missing_mean()
    cleaner.standardize_zscore(['feature_a', 'feature_b'])
    
    cleaned_df = cleaner.get_cleaned_data()
    print(f"Cleaned data shape: {cleaned_df.shape}")
    print(f"Removed {cleaner.get_removed_count()} outliers")
    print(f"Cleaned data summary:\n{cleaned_df.describe()}")
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
            print(f"Data loaded successfully. Shape: {self.df.shape}")
            return True
        except FileNotFoundError:
            print(f"Error: File not found at {self.file_path}")
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def check_missing_values(self):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return None
        
        missing_counts = self.df.isnull().sum()
        missing_percent = (missing_counts / len(self.df)) * 100
        
        missing_report = pd.DataFrame({
            'missing_count': missing_counts,
            'missing_percent': missing_percent
        })
        
        return missing_report[missing_report['missing_count'] > 0]
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return False
        
        if columns is None:
            columns = self.df.columns
        
        numeric_cols = self.df[columns].select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0]
                else:
                    fill_value = 0
                
                self.df[col].fillna(fill_value, inplace=True)
                print(f"Filled missing values in '{col}' with {strategy}: {fill_value}")
        
        return True
    
    def save_cleaned_data(self, output_path=None):
        if self.df is None:
            print("No data to save. Call load_data() first.")
            return False
        
        if output_path is None:
            output_path = self.file_path.parent / f"cleaned_{self.file_path.name}"
        
        try:
            self.df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
            return True
        except Exception as e:
            print(f"Error saving data: {e}")
            return False

def process_csv_file(input_file, output_file=None):
    cleaner = DataCleaner(input_file)
    
    if not cleaner.load_data():
        return False
    
    missing_report = cleaner.check_missing_values()
    if missing_report is not None and not missing_report.empty:
        print("Missing values found:")
        print(missing_report)
        
        cleaner.handle_missing_values(strategy='median')
    else:
        print("No missing values found.")
    
    if output_file:
        return cleaner.save_cleaned_data(output_file)
    else:
        return cleaner.save_cleaned_data()

if __name__ == "__main__":
    sample_file = "sample_data.csv"
    result = process_csv_file(sample_file)
    
    if result:
        print("Data cleaning completed successfully.")
    else:
        print("Data cleaning failed.")