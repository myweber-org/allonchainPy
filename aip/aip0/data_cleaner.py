import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
    return cleaned_df

def validate_data(df, required_columns):
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    return Trueimport pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a specified column in a DataFrame using the IQR method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing summary statistics.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def process_dataframe(df, column):
    """
    Main function to process DataFrame by removing outliers and calculating statistics.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    tuple: (cleaned_df, original_stats, cleaned_stats)
    """
    original_stats = calculate_summary_statistics(df, column)
    cleaned_df = remove_outliers_iqr(df, column)
    cleaned_stats = calculate_summary_statistics(cleaned_df, column)
    
    return cleaned_df, original_stats, cleaned_stats

if __name__ == "__main__":
    sample_data = {
        'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 100, 12, 14, 13, 12, 11, 14, 13, 12, 14, 15]
    }
    df = pd.DataFrame(sample_data)
    
    print("Original DataFrame:")
    print(df)
    print("\nProcessing 'values' column...")
    
    try:
        cleaned_df, original_stats, cleaned_stats = process_dataframe(df, 'values')
        
        print(f"\nOriginal statistics: {original_stats}")
        print(f"Cleaned statistics: {cleaned_stats}")
        print(f"\nRows removed: {len(df) - len(cleaned_df)}")
        print("\nCleaned DataFrame:")
        print(cleaned_df)
        
    except ValueError as e:
        print(f"Error: {e}")
def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, data):
        self.data = data
        self.original_shape = data.shape
        
    def remove_outliers_iqr(self, columns=None, factor=1.5):
        if columns is None:
            columns = self.data.columns
            
        clean_data = self.data.copy()
        for col in columns:
            if pd.api.types.is_numeric_dtype(clean_data[col]):
                Q1 = clean_data[col].quantile(0.25)
                Q3 = clean_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                clean_data = clean_data[(clean_data[col] >= lower_bound) & (clean_data[col] <= upper_bound)]
        return clean_data
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        normalized_data = self.data.copy()
        for col in columns:
            if pd.api.types.is_numeric_dtype(normalized_data[col]):
                min_val = normalized_data[col].min()
                max_val = normalized_data[col].max()
                if max_val != min_val:
                    normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
        return normalized_data
    
    def standardize_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        standardized_data = self.data.copy()
        for col in columns:
            if pd.api.types.is_numeric_dtype(standardized_data[col]):
                z_scores = np.abs(stats.zscore(standardized_data[col]))
                standardized_data = standardized_data[z_scores < threshold]
        return standardized_data
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        filled_data = self.data.copy()
        for col in columns:
            if filled_data[col].isnull().any():
                if strategy == 'mean':
                    fill_value = filled_data[col].mean()
                elif strategy == 'median':
                    fill_value = filled_data[col].median()
                elif strategy == 'mode':
                    fill_value = filled_data[col].mode()[0]
                else:
                    fill_value = 0
                filled_data[col] = filled_data[col].fillna(fill_value)
        return filled_data
    
    def get_cleaning_report(self):
        report = {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'current_rows': self.data.shape[0],
            'current_columns': self.data.shape[1],
            'rows_removed': self.original_shape[0] - self.data.shape[0],
            'missing_values': self.data.isnull().sum().sum()
        }
        return report

def load_and_clean_dataset(filepath, cleaning_steps=None):
    df = pd.read_csv(filepath)
    cleaner = DataCleaner(df)
    
    if cleaning_steps is None:
        cleaning_steps = ['missing', 'outliers', 'normalize']
    
    for step in cleaning_steps:
        if step == 'missing':
            df = cleaner.handle_missing_values(strategy='mean')
        elif step == 'outliers':
            df = cleaner.remove_outliers_iqr()
        elif step == 'normalize':
            df = cleaner.normalize_minmax()
        elif step == 'standardize':
            df = cleaner.standardize_zscore()
        cleaner.data = df
    
    return df, cleaner.get_cleaning_report()