
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, threshold=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_clean = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                mask = (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)
                df_clean = df_clean[mask]
                
        removed_count = self.original_shape[0] - df_clean.shape[0]
        self.df = df_clean.reset_index(drop=True)
        return removed_count
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_normalized = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    df_normalized[col] = (self.df[col] - min_val) / (max_val - min_val)
                    
        self.df = df_normalized
        return self
    
    def standardize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_standardized = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    df_standardized[col] = (self.df[col] - mean_val) / std_val
                    
        self.df = df_standardized
        return self
    
    def fill_missing_mean(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_filled = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                mean_val = self.df[col].mean()
                df_filled[col] = self.df[col].fillna(mean_val)
                
        self.df = df_filled
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'cleaned_rows': self.df.shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'missing_values': self.df.isnull().sum().sum()
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 1, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    
    indices = np.random.choice(df.index, size=50, replace=False)
    for idx in indices:
        df.loc[idx, 'feature_a'] = np.random.uniform(300, 500)
        
    nan_indices = np.random.choice(df.index, size=30, replace=False)
    df.loc[nan_indices, 'feature_b'] = np.nan
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    print("Sample data created with shape:", sample_df.shape)
    
    cleaner = DataCleaner(sample_df)
    print("\nInitial summary:")
    print(cleaner.get_summary())
    
    outliers_removed = cleaner.remove_outliers_iqr(['feature_a', 'feature_b'])
    print(f"\nRemoved {outliers_removed} outliers using IQR method")
    
    cleaner.fill_missing_mean(['feature_b'])
    print("Filled missing values with mean")
    
    cleaner.normalize_minmax(['feature_a', 'feature_b', 'feature_c'])
    print("Applied min-max normalization")
    
    print("\nFinal summary:")
    print(cleaner.get_summary())
    
    cleaned_df = cleaner.get_cleaned_data()
    print(f"\nCleaned data shape: {cleaned_df.shape}")
    print("\nFirst 5 rows of cleaned data:")
    print(cleaned_df.head())
def remove_duplicates_preserve_order(iterable):
    seen = set()
    result = []
    for item in iterable:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport pandas as pd
import numpy as np

def clean_csv_data(file_path, fill_strategy='mean', output_path=None):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Args:
        file_path (str): Path to input CSV file
        fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'zero')
        output_path (str, optional): Path for cleaned output CSV. If None, returns DataFrame
    
    Returns:
        pd.DataFrame or None: Cleaned DataFrame if output_path is None
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data with shape: {df.shape}")
        
        # Remove duplicate rows
        initial_count = len(df)
        df.drop_duplicates(inplace=True)
        duplicates_removed = initial_count - len(df)
        print(f"Removed {duplicates_removed} duplicate rows")
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                print(f"Column '{col}' has {missing_count} missing values")
                
                if col in numeric_cols:
                    if fill_strategy == 'mean':
                        fill_value = df[col].mean()
                    elif fill_strategy == 'median':
                        fill_value = df[col].median()
                    elif fill_strategy == 'zero':
                        fill_value = 0
                    else:
                        fill_value = df[col].mean()
                    df[col].fillna(fill_value, inplace=True)
                    
                elif col in categorical_cols:
                    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
        # Reset index after cleaning
        df.reset_index(drop=True, inplace=True)
        print(f"Final data shape: {df.shape}")
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
            return None
        else:
            return df
            
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of required column names
    
    Returns:
        bool: True if validation passes
    """
    if df is None or df.empty:
        print("Validation failed: DataFrame is empty or None")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.any(np.isinf(df[col])):
            print(f"Validation warning: Column '{col}' contains infinite values")
    
    print("Data validation passed")
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10.5, None, 15.2, 20.1, None, 10.5],
        'category': ['A', 'B', None, 'A', 'C', 'A']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', fill_strategy='mean')
    
    if cleaned_df is not None:
        is_valid = validate_dataframe(cleaned_df, required_columns=['id', 'value', 'category'])
        print(f"Data validation result: {is_valid}")
        print("\nCleaned data preview:")
        print(cleaned_df.head())