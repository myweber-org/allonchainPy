
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, column, multiplier=1.5):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
        return self
        
    def remove_outliers_zscore(self, column, threshold=3):
        z_scores = np.abs(stats.zscore(self.df[column]))
        self.df = self.df[z_scores < threshold]
        return self
        
    def normalize_column(self, column, method='minmax'):
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
        elif method == 'standard':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            self.df[column] = (self.df[column] - mean_val) / std_val
        return self
        
    def fill_missing(self, column, method='mean'):
        if method == 'mean':
            fill_value = self.df[column].mean()
        elif method == 'median':
            fill_value = self.df[column].median()
        elif method == 'mode':
            fill_value = self.df[column].mode()[0]
        else:
            fill_value = method
            
        self.df[column] = self.df[column].fillna(fill_value)
        return self
        
    def get_cleaned_data(self):
        return self.df
        
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_strategy='mean', drop_threshold=0.5):
    """
    Load and clean CSV data by handling missing values.
    
    Args:
        filepath: Path to the CSV file
        fill_strategy: Strategy for filling missing values ('mean', 'median', 'mode', 'zero')
        drop_threshold: Drop columns with missing values exceeding this ratio (0.0 to 1.0)
    
    Returns:
        Cleaned DataFrame and cleaning report dictionary
    """
    try:
        df = pd.read_csv(filepath)
        original_shape = df.shape
        cleaning_report = {
            'original_rows': original_shape[0],
            'original_columns': original_shape[1],
            'missing_values': df.isnull().sum().sum(),
            'dropped_columns': []
        }
        
        # Drop columns with too many missing values
        missing_ratio = df.isnull().sum() / len(df)
        columns_to_drop = missing_ratio[missing_ratio > drop_threshold].index.tolist()
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            cleaning_report['dropped_columns'] = columns_to_drop
        
        # Fill remaining missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        
        if fill_strategy == 'mean' and len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif fill_strategy == 'median' and len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif fill_strategy == 'zero' and len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # For categorical columns, fill with mode
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_value = df[col].mode()
                if not mode_value.empty:
                    df[col] = df[col].fillna(mode_value.iloc[0])
        
        cleaning_report['final_rows'] = df.shape[0]
        cleaning_report['final_columns'] = df.shape[1]
        cleaning_report['remaining_missing'] = df.isnull().sum().sum()
        
        return df, cleaning_report
        
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except pd.errors.EmptyDataError:
        raise ValueError("The CSV file is empty")
    except Exception as e:
        raise RuntimeError(f"Error cleaning data: {str(e)}")

def validate_dataframe(df, required_columns=None, unique_constraints=None):
    """
    Validate DataFrame structure and constraints.
    
    Args:
        df: DataFrame to validate
        required_columns: List of columns that must be present
        unique_constraints: List of columns that should have unique values
    
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Missing required columns: {missing_cols}")
    
    if unique_constraints:
        for col in unique_constraints:
            if col in df.columns:
                duplicates = df[col].duplicated().sum()
                if duplicates > 0:
                    validation_result['warnings'].append(
                        f"Column '{col}' has {duplicates} duplicate values"
                    )
    
    if df.empty:
        validation_result['is_valid'] = False
        validation_result['errors'].append("DataFrame is empty")
    
    return validation_result

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5],
        'value': [10.5, None, 15.2, None, 20.1],
        'category': ['A', 'B', None, 'A', 'C'],
        'score': [100, 200, 150, None, 180]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df, report = clean_csv_data('test_data.csv', fill_strategy='mean')
    print(f"Cleaning report: {report}")
    print(f"Cleaned data shape: {cleaned_df.shape}")
    
    import os
    os.remove('test_data.csv')
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    drop_duplicates (bool): Whether to drop duplicate rows
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
        print(f"Removed {len(df) - len(cleaned_df)} duplicate rows")
    
    if cleaned_df.isnull().sum().sum() > 0:
        print(f"Found {cleaned_df.isnull().sum().sum()} missing values")
        
        if fill_missing == 'drop':
            cleaned_df = cleaned_df.dropna()
            print("Dropped rows with missing values")
        elif fill_missing in ['mean', 'median']:
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if fill_missing == 'mean':
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                else:
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            print(f"Filled missing numeric values with {fill_missing}")
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
            print("Filled missing values with mode")
    
    print(f"Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return True

def remove_outliers_iqr(df, columns=None, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range (IQR) method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): Columns to check for outliers (None for all numeric columns)
    multiplier (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    original_len = len(df)
    mask = pd.Series([True] * len(df))
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            col_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            mask = mask & col_mask
    
    cleaned_df = df[mask].copy()
    outliers_removed = original_len - len(cleaned_df)
    
    if outliers_removed > 0:
        print(f"Removed {outliers_removed} outliers using IQR method")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 3, 4, 5, None, 7, 8, 9, 100],
        'B': [10, 20, 20, 30, 40, 50, 60, 70, 80, 90, 1000],
        'C': ['a', 'b', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nDataFrame validation: {is_valid}")
    
    final_df = remove_outliers_iqr(cleaned, columns=['A', 'B'])
    print(f"\nFinal DataFrame shape after outlier removal: {final_df.shape}")