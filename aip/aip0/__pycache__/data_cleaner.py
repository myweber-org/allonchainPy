
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
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, columns=None):
    """
    Remove outliers from DataFrame using Interquartile Range method.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to process, if None processes all numeric columns
    
    Returns:
        DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = df.copy()
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize data using specified method.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to normalize
        method: 'minmax' or 'zscore'
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_normalized = df.copy()
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            if method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val != min_val:
                    df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
            
            elif method == 'zscore':
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val != 0:
                    df_normalized[col] = (df[col] - mean_val) / std_val
    
    return df_normalized

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        df: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'drop'
        columns: list of column names to process
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = df.columns.tolist()
    
    df_handled = df.copy()
    
    for col in columns:
        if col in df.columns and df[col].isnull().any():
            if strategy == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
                df_handled[col].fillna(df[col].mean(), inplace=True)
            
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[col]):
                df_handled[col].fillna(df[col].median(), inplace=True)
            
            elif strategy == 'mode':
                df_handled[col].fillna(df[col].mode()[0], inplace=True)
            
            elif strategy == 'drop':
                df_handled = df_handled.dropna(subset=[col])
    
    return df_handled.reset_index(drop=True)
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
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

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
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
        'count': len(df[column])
    }
    
    return stats

def clean_dataset(df, columns_to_clean):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for column in columns_to_clean:
        if column in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 100, 27, 28, 29, -10],
        'humidity': [45, 46, 47, 48, 49, 200, 50, 51, 52, -5],
        'pressure': [1013, 1014, 1015, 1016, 1017, 2000, 1018, 1019, 1020, 500]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\nSummary statistics before cleaning:")
    for col in df.columns:
        print(f"{col}: {calculate_summary_statistics(df, col)}")
    
    cleaned_df = clean_dataset(df, ['temperature', 'humidity', 'pressure'])
    print("\nCleaned dataset:")
    print(cleaned_df)
    print("\nSummary statistics after cleaning:")
    for col in cleaned_df.columns:
        print(f"{col}: {calculate_summary_statistics(cleaned_df, col)}")