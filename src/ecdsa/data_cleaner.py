
import pandas as pd

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
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            elif fill_missing == 'median':
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None, inplace=True)
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (bool, str) indicating validation success and message.
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"import numpy as np
import pandas as pd
from scipy import stats

def normalize_data(data, method='zscore'):
    if method == 'zscore':
        return (data - np.mean(data)) / np.std(data)
    elif method == 'minmax':
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    else:
        raise ValueError("Method must be 'zscore' or 'minmax'")

def remove_outliers_iqr(data, threshold=1.5):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return data[(data >= lower_bound) & (data <= upper_bound)]

def clean_dataset(df, column, normalize_method='zscore', outlier_threshold=1.5):
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in DataFrame")
    
    original_data = df[column].dropna().values
    cleaned_data = remove_outliers_iqr(original_data, outlier_threshold)
    normalized_data = normalize_data(cleaned_data, normalize_method)
    
    result_df = pd.DataFrame({
        'original': original_data,
        'cleaned': np.concatenate([cleaned_data, [np.nan] * (len(original_data) - len(cleaned_data))]),
        'normalized': np.concatenate([normalized_data, [np.nan] * (len(original_data) - len(normalized_data))])
    })
    
    return result_df

def calculate_statistics(data):
    return {
        'mean': np.mean(data),
        'std': np.std(data),
        'median': np.median(data),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data)
    }
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
    
    return filtered_df

def calculate_summary_statistics(df):
    """
    Calculate summary statistics for numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    pd.DataFrame: Summary statistics
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary = df[numeric_cols].describe()
    
    return summary

def clean_missing_values(df, strategy='mean'):
    """
    Handle missing values in numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Strategy for imputation ('mean', 'median', 'mode', 'drop')
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    df_clean = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if df[col].isnull().any():
            if strategy == 'mean':
                df_clean[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'median':
                df_clean[col].fillna(df[col].median(), inplace=True)
            elif strategy == 'mode':
                df_clean[col].fillna(df[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                df_clean = df_clean.dropna(subset=[col])
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
    
    return df_clean

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, 3, 4, 5, 100, 7, 8, 9, 10],
        'B': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'C': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = remove_outliers_iqr(df, 'A')
    print("\nDataFrame after removing outliers from column 'A':")
    print(cleaned_df)
    
    stats = calculate_summary_statistics(df)
    print("\nSummary statistics:")
    print(stats)
import pandas as pd
import numpy as np
from pathlib import Path

class DataCleaner:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.df = None
        
    def load_data(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        if self.file_path.suffix == '.csv':
            self.df = pd.read_csv(self.file_path)
        elif self.file_path.suffix in ['.xlsx', '.xls']:
            self.df = pd.read_excel(self.file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel files.")
        
        print(f"Loaded data with shape: {self.df.shape}")
        return self
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns:
                if self.df[col].isnull().any():
                    if strategy == 'mean':
                        fill_value = self.df[col].mean()
                    elif strategy == 'median':
                        fill_value = self.df[col].median()
                    elif strategy == 'mode':
                        fill_value = self.df[col].mode()[0]
                    elif strategy == 'drop':
                        self.df = self.df.dropna(subset=[col])
                        continue
                    else:
                        raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")
                    
                    self.df[col].fillna(fill_value, inplace=True)
                    print(f"Filled missing values in column '{col}' using {strategy} strategy")
        
        return self
    
    def remove_duplicates(self, subset=None, keep='first'):
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        removed_count = initial_count - len(self.df)
        
        print(f"Removed {removed_count} duplicate rows")
        return self
    
    def normalize_numeric_columns(self, columns=None):
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns:
                if self.df[col].std() != 0:
                    self.df[col] = (self.df[col] - self.df[col].mean()) / self.df[col].std()
                    print(f"Normalized column '{col}'")
        
        return self
    
    def save_cleaned_data(self, output_path=None):
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if output_path is None:
            output_path = self.file_path.parent / f"cleaned_{self.file_path.name}"
        
        output_path = Path(output_path)
        
        if output_path.suffix == '.csv':
            self.df.to_csv(output_path, index=False)
        elif output_path.suffix in ['.xlsx', '.xls']:
            self.df.to_excel(output_path, index=False)
        else:
            raise ValueError("Unsupported output format. Use CSV or Excel files.")
        
        print(f"Saved cleaned data to: {output_path}")
        return output_path
    
    def get_summary(self):
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        summary = {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().sum(),
            'duplicates': len(self.df) - len(self.df.drop_duplicates()),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns)
        }
        
        return summary

def clean_dataset(input_file, output_file=None, missing_strategy='mean'):
    cleaner = DataCleaner(input_file)
    
    try:
        cleaner.load_data()
        cleaner.handle_missing_values(strategy=missing_strategy)
        cleaner.remove_duplicates()
        cleaner.normalize_numeric_columns()
        
        if output_file:
            cleaner.save_cleaned_data(output_file)
        else:
            cleaner.save_cleaned_data()
        
        summary = cleaner.get_summary()
        print(f"Data cleaning completed. Summary: {summary}")
        
        return cleaner.df
        
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        raise

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5, 5],
        'B': [10, 20, 30, np.nan, 50, 50],
        'C': ['a', 'b', 'c', 'd', 'e', 'e']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_dataset('test_data.csv', 'cleaned_test_data.csv')
    print("Sample data cleaning completed successfully")
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

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            original_len = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_len - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{col}'")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    bool: True if validation passes
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True

if __name__ == "__main__":
    sample_data = {
        'id': range(1, 101),
        'value': np.random.randn(100) * 10 + 50,
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[10, 'value'] = 200
    df.loc[20, 'value'] = -100
    
    print("Original DataFrame shape:", df.shape)
    print("Original statistics:")
    print(df['value'].describe())
    
    cleaned_df = clean_numeric_data(df, ['value'])
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned statistics:")
    print(cleaned_df['value'].describe())