import pandas as pd

def clean_dataset(df, drop_duplicates=True, fillna_strategy='mean'):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows.
        fillna_strategy (str): Strategy for filling NaN values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fillna_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fillna_strategy in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fillna_strategy == 'mean':
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            elif fillna_strategy == 'median':
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    elif fillna_strategy == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None, inplace=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
    Returns:
        dict: Validation results with keys 'is_valid' and 'message'.
    """
    validation_result = {'is_valid': True, 'message': 'Validation passed'}
    
    if df.empty:
        validation_result['is_valid'] = False
        validation_result['message'] = 'DataFrame is empty'
        return validation_result
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_result['is_valid'] = False
            validation_result['message'] = f'Missing required columns: {missing_columns}'
    
    return validation_result

if __name__ == '__main__':
    sample_data = {
        'A': [1, 2, 2, 3, None],
        'B': [4, None, 6, 6, 8],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fillna_strategy='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    validation = validate_dataframe(cleaned, required_columns=['A', 'B'])
    print(f"\nValidation: {validation['message']}")
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def detect_outliers_iqr(self, column):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        return outliers
    
    def remove_outliers_zscore(self, column, threshold=3):
        z_scores = np.abs(stats.zscore(self.df[column].dropna()))
        mask = z_scores < threshold
        self.df = self.df[mask]
        return self.df
    
    def normalize_column(self, column, method='minmax'):
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            self.df[f'{column}_normalized'] = (self.df[column] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            self.df[f'{column}_normalized'] = (self.df[column] - mean_val) / std_val
        return self.df
    
    def handle_missing_values(self, strategy='mean'):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
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
                
                self.df[col] = self.df[col].fillna(fill_value)
        
        return self.df
    
    def get_cleaning_report(self):
        cleaned_shape = self.df.shape
        rows_removed = self.original_shape[0] - cleaned_shape[0]
        cols_added = cleaned_shape[1] - self.original_shape[1]
        
        report = {
            'original_rows': self.original_shape[0],
            'cleaned_rows': cleaned_shape[0],
            'rows_removed': rows_removed,
            'columns_added': cols_added,
            'missing_values': self.df.isnull().sum().sum()
        }
        
        return report
    
    def save_cleaned_data(self, filepath):
        self.df.to_csv(filepath, index=False)
        return f"Data saved to {filepath}"

def example_usage():
    data = {
        'age': [25, 30, 35, 200, 28, 32, 150, 29, 31, None],
        'salary': [50000, 60000, 55000, 1000000, 52000, 58000, 900000, 54000, 56000, 51000],
        'score': [85, 92, 78, 99, 88, 91, 30, 86, 89, 84]
    }
    
    df = pd.DataFrame(data)
    cleaner = DataCleaner(df)
    
    print("Original data:")
    print(df)
    print("\nDetecting outliers in 'salary':")
    print(cleaner.detect_outliers_iqr('salary'))
    
    cleaner.remove_outliers_zscore('salary')
    cleaner.handle_missing_values(strategy='mean')
    cleaner.normalize_column('score', method='minmax')
    
    print("\nCleaned data:")
    print(cleaner.df)
    
    report = cleaner.get_cleaning_report()
    print("\nCleaning report:")
    for key, value in report.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    example_usage()