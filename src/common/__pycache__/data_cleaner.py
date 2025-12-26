import pandas as pd
import numpy as np
from typing import Optional, Dict, List

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def handle_missing_values(self, strategy: str = 'mean', columns: Optional[List[str]] = None) -> 'DataCleaner':
        if columns is None:
            columns = self.df.columns
            
        for col in columns:
            if col in self.df.columns:
                if strategy == 'mean' and pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif strategy == 'median' and pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                elif strategy == 'mode':
                    self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else None, inplace=True)
                elif strategy == 'drop':
                    self.df = self.df.dropna(subset=[col])
                else:
                    self.df[col].fillna('Unknown', inplace=True)
        return self
    
    def convert_types(self, type_mapping: Dict[str, str]) -> 'DataCleaner':
        for col, dtype in type_mapping.items():
            if col in self.df.columns:
                try:
                    if dtype == 'datetime':
                        self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    elif dtype == 'numeric':
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    elif dtype == 'category':
                        self.df[col] = self.df[col].astype('category')
                    else:
                        self.df[col] = self.df[col].astype(dtype)
                except Exception as e:
                    print(f"Warning: Could not convert column {col} to {dtype}: {e}")
        return self
    
    def remove_outliers(self, column: str, method: str = 'iqr', threshold: float = 1.5) -> 'DataCleaner':
        if column not in self.df.columns or not pd.api.types.is_numeric_dtype(self.df[column]):
            return self
            
        if method == 'iqr':
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
            
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(self.df[column].dropna()))
            self.df = self.df[z_scores < threshold]
            
        return self
    
    def normalize_column(self, column: str, method: str = 'minmax') -> 'DataCleaner':
        if column not in self.df.columns or not pd.api.types.is_numeric_dtype(self.df[column]):
            return self
            
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            if max_val != min_val:
                self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
                
        elif method == 'standard':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            if std_val != 0:
                self.df[column] = (self.df[column] - mean_val) / std_val
                
        return self
    
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df
    
    def get_cleaning_report(self) -> Dict:
        return {
            'original_shape': self.original_shape,
            'cleaned_shape': self.df.shape,
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'columns_removed': self.original_shape[1] - self.df.shape[1],
            'missing_values_remaining': self.df.isnull().sum().sum(),
            'data_types': self.df.dtypes.to_dict()
        }

def clean_csv_file(input_path: str, output_path: str, cleaning_steps: Dict) -> Dict:
    try:
        df = pd.read_csv(input_path)
        cleaner = DataCleaner(df)
        
        if 'handle_missing' in cleaning_steps:
            cleaner.handle_missing_values(**cleaning_steps['handle_missing'])
            
        if 'convert_types' in cleaning_steps:
            cleaner.convert_types(cleaning_steps['convert_types'])
            
        if 'remove_outliers' in cleaning_steps:
            for outlier_config in cleaning_steps['remove_outliers']:
                cleaner.remove_outliers(**outlier_config)
                
        if 'normalize' in cleaning_steps:
            for normalize_config in cleaning_steps['normalize']:
                cleaner.normalize_column(**normalize_config)
        
        cleaned_df = cleaner.get_cleaned_data()
        cleaned_df.to_csv(output_path, index=False)
        
        return {
            'success': True,
            'report': cleaner.get_cleaning_report(),
            'output_file': output_path
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'output_file': None
        }import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            elif fill_missing == 'median':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for required columns and data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    tuple: (bool, str) indicating validation result and message.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    return True, "Dataset validation passed"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 30, 40, 50],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_dataset(cleaned, required_columns=['A', 'B'])
    print(f"\nValidation: {is_valid}, Message: {message}")