
import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None, missing_strategy='mean'):
    """
    Load a CSV file, clean missing values, and optionally save cleaned data.
    
    Parameters:
    file_path (str): Path to input CSV file
    output_path (str, optional): Path to save cleaned CSV file
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'drop', 'zero')
    
    Returns:
    pandas.DataFrame: Cleaned dataframe
    """
    
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data with shape: {df.shape}")
        
        # Check for missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values")
            
            # Handle missing values based on strategy
            if missing_strategy == 'mean':
                df = df.fillna(df.mean(numeric_only=True))
            elif missing_strategy == 'median':
                df = df.fillna(df.median(numeric_only=True))
            elif missing_strategy == 'drop':
                df = df.dropna()
            elif missing_strategy == 'zero':
                df = df.fillna(0)
            else:
                raise ValueError(f"Unknown strategy: {missing_strategy}")
            
            print(f"Missing values handled using '{missing_strategy}' strategy")
        
        # Remove duplicate rows
        initial_rows = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate rows")
        
        # Convert column names to lowercase with underscores
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # Save cleaned data if output path provided
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Parameters:
    df (pandas.DataFrame): Dataframe to validate
    required_columns (list, optional): List of required column names
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    
    if df is None or df.empty:
        print("Validation failed: Dataframe is empty or None")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Validation failed: Missing required columns: {missing_columns}")
            return False
    
    # Check for infinite values
    if df.select_dtypes(include=[np.number]).applymap(np.isinf).any().any():
        print("Warning: Dataframe contains infinite values")
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'Name': ['Alice', 'Bob', 'Charlie', None, 'Eve'],
        'Age': [25, 30, None, 35, 40],
        'Score': [85.5, 92.0, 78.5, None, 88.0]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', 'cleaned_sample.csv', 'mean')
    
    if cleaned_df is not None:
        print(f"Cleaned dataframe shape: {cleaned_df.shape}")
        print(cleaned_df.head())
        
        if validate_dataframe(cleaned_df, ['name', 'age', 'score']):
            print("Data validation passed")
        else:
            print("Data validation failed")
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
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

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def main():
    # Example usage
    data = {
        'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 100, 12, 13, 12, 11, 10, 14, 13, 12, 11, 10]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print(f"\nOriginal shape: {df.shape}")
    
    # Remove outliers
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print(f"\nCleaned shape: {cleaned_df.shape}")
    
    # Calculate statistics
    original_stats = calculate_summary_stats(df, 'values')
    cleaned_stats = calculate_summary_stats(cleaned_df, 'values')
    
    print("\nOriginal statistics:")
    for key, value in original_stats.items():
        print(f"{key}: {value}")
    
    print("\nCleaned statistics:")
    for key, value in cleaned_stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
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
        
        self.df = df_clean.reset_index(drop=True)
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
        
        return self

    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)
        
        return self

    def get_cleaned_data(self):
        return self.df

    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]

def process_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        cleaner = DataCleaner(df)
        
        cleaner.fill_missing_median()
        cleaner.remove_outliers_iqr(threshold=1.5)
        cleaner.normalize_minmax()
        
        cleaned_df = cleaner.get_cleaned_data()
        removed = cleaner.get_removed_count()
        
        print(f"Original rows: {df.shape[0]}")
        print(f"Cleaned rows: {cleaned_df.shape[0]}")
        print(f"Rows removed: {removed}")
        
        return cleaned_df
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return None