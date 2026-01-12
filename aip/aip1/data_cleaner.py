
import pandas as pd
import numpy as np

def clean_dataset(df, column_mapping=None, drop_duplicates=True, fill_missing=True):
    """
    Clean a pandas DataFrame by standardizing columns, removing duplicates,
    and handling missing values.
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df.rename(columns=column_mapping, inplace=True)
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df.drop_duplicates(inplace=True)
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing:
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype in ['int64', 'float64']:
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
            elif cleaned_df[col].dtype == 'object':
                cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else '', inplace=True)
    
    cleaned_df.reset_index(drop=True, inplace=True)
    return cleaned_df

def validate_data(df, required_columns=None, numeric_columns=None):
    """
    Validate the cleaned dataset for required columns and numeric data integrity.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if numeric_columns:
        for col in numeric_columns:
            if col in df.columns:
                if df[col].dtype not in ['int64', 'float64']:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        print(f"Warning: Column {col} could not be converted to numeric")
    
    return True

def export_cleaned_data(df, output_path, format='csv'):
    """
    Export the cleaned DataFrame to a file.
    """
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'excel':
        df.to_excel(output_path, index=False)
    elif format == 'json':
        df.to_json(output_path, orient='records')
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Cleaned data exported to {output_path}")

if __name__ == "__main__":
    sample_data = {
        'Name': ['Alice', 'Bob', 'Alice', 'Charlie', None],
        'Age': [25, 30, 25, None, 35],
        'Salary': [50000, 60000, 50000, 70000, 80000],
        'Department': ['HR', 'IT', 'HR', 'Finance', 'IT']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df, column_mapping=None, drop_duplicates=True, fill_missing=True)
    print("Cleaned dataset:")
    print(cleaned)
    
    try:
        validate_data(cleaned, required_columns=['Name', 'Age', 'Salary'])
        print("\nData validation passed")
    except ValueError as e:
        print(f"\nData validation failed: {e}")import pandas as pd

def clean_dataset(df, remove_duplicates=True, fill_method='drop'):
    """
    Clean a pandas DataFrame by handling missing values and optionally removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    remove_duplicates (bool): If True, remove duplicate rows.
    fill_method (str): Method to handle missing values: 'drop' to remove rows, 
                       'ffill' to forward fill, 'bfill' to backward fill.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_method == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_method == 'ffill':
        cleaned_df = cleaned_df.ffill()
    elif fill_method == 'bfill':
        cleaned_df = cleaned_df.bfill()
    else:
        raise ValueError("fill_method must be 'drop', 'ffill', or 'bfill'")
    
    # Remove duplicates if requested
    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    dict: Dictionary with validation results.
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'null_count': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_results['missing_columns'] = missing_columns
        validation_results['all_required_columns_present'] = len(missing_columns) == 0
    
    return validation_results

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'A': [1, 2, None, 4, 5],
        'B': [5, 6, 7, None, 9],
        'C': [10, 11, 12, 13, 10]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation results:")
    print(validate_dataset(df))
    
    cleaned = clean_dataset(df, remove_duplicates=True, fill_method='ffill')
    print("\nCleaned DataFrame:")
    print(cleaned)
    print("\nValidation results after cleaning:")
    print(validate_dataset(cleaned))import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Method to fill missing values. Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
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
        if fill_missing == 'mean':
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
        else:
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                mode_val = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None
                cleaned_df[col] = cleaned_df[col].fillna(mode_val)
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if df.empty:
        print("Validation failed: DataFrame is empty.")
        return False
    
    if len(df) < min_rows:
        print(f"Validation failed: DataFrame has fewer than {min_rows} rows.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, None, 4, 1],
        'B': [5, None, 7, 8, 5],
        'C': ['x', 'y', 'z', None, 'x']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_data(cleaned, required_columns=['A', 'B'], min_rows=3)
    print(f"\nData validation result: {is_valid}")import numpy as np
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
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                mask = (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)
                df_clean = df_clean[mask]
                
        self.df = df_clean.reset_index(drop=True)
        removed_count = self.original_shape[0] - self.df.shape[0]
        print(f"Removed {removed_count} outliers using IQR method")
        return self
        
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
                    
        print(f"Applied min-max normalization to {len(columns)} columns")
        return self
        
    def standardize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
                    
        print(f"Applied z-score standardization to {len(columns)} columns")
        return self
        
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns and self.df[col].isnull().any():
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
                
        print(f"Filled missing values with median for {len(columns)} columns")
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
    df.loc[np.random.choice(df.index, 50), 'feature_a'] = np.nan
    df.loc[np.random.choice(df.index, 20), 'feature_b'] = 1000
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    print("Original data shape:", sample_df.shape)
    
    cleaner = DataCleaner(sample_df)
    cleaned_df = (cleaner
                 .fill_missing_median()
                 .remove_outliers_iqr(['feature_a', 'feature_b'])
                 .standardize_zscore(['feature_a', 'feature_b'])
                 .normalize_minmax(['feature_c'])
                 .get_cleaned_data())
    
    print("Cleaned data shape:", cleaned_df.shape)
    print("\nSummary:")
    for key, value in cleaner.get_summary().items():
        print(f"{key}: {value}")
    
    print("\nFirst 5 rows of cleaned data:")
    print(cleaned_df.head())
def deduplicate_list(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result