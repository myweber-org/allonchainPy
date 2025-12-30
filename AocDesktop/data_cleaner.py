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
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to remove duplicate rows
        fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if cleaned_df.isnull().sum().sum() > 0:
        print(f"Found {cleaned_df.isnull().sum().sum()} missing values")
        
        if fill_missing == 'drop':
            cleaned_df = cleaned_df.dropna()
            print("Dropped rows with missing values")
        elif fill_missing == 'mean':
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
            print("Filled missing numeric values with column means")
        elif fill_missing == 'median':
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
            print("Filled missing numeric values with column medians")
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    mode_val = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown'
                    cleaned_df[col] = cleaned_df[col].fillna(mode_val)
            print("Filled missing categorical values with column modes")
    
    print(f"Data cleaning complete. Final shape: {cleaned_df.shape}")
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of column names that must be present
        min_rows (int): Minimum number of rows required
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if df.empty:
        print("Validation failed: DataFrame is empty")
        return False
    
    if len(df) < min_rows:
        print(f"Validation failed: DataFrame has only {len(df)} rows, minimum required is {min_rows}")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    print("Data validation passed")
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', None, 'Eve', 'Frank'],
        'age': [25, 30, 30, 35, None, 40],
        'score': [85.5, 92.0, 92.0, 78.5, 88.0, 95.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    validation_result = validate_data(cleaned, required_columns=['id', 'name', 'age'], min_rows=3)
    print(f"\nValidation result: {validation_result}")