
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
        
        clean_df = self.df.copy()
        for col in columns:
            if col in clean_df.columns:
                Q1 = clean_df[col].quantile(0.25)
                Q3 = clean_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                clean_df = clean_df[(clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)]
        
        self.df = clean_df
        return self
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        clean_df = self.df.copy()
        for col in columns:
            if col in clean_df.columns:
                z_scores = np.abs(stats.zscore(clean_df[col].dropna()))
                clean_df = clean_df[(z_scores < threshold) | clean_df[col].isna()]
        
        self.df = clean_df
        return self
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        normalized_df = self.df.copy()
        for col in columns:
            if col in normalized_df.columns:
                col_min = normalized_df[col].min()
                col_max = normalized_df[col].max()
                if col_max != col_min:
                    normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
        
        self.df = normalized_df
        return self
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        normalized_df = self.df.copy()
        for col in columns:
            if col in normalized_df.columns:
                col_mean = normalized_df[col].mean()
                col_std = normalized_df[col].std()
                if col_std > 0:
                    normalized_df[col] = (normalized_df[col] - col_mean) / col_std
        
        self.df = normalized_df
        return self
    
    def fill_missing_mean(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        filled_df = self.df.copy()
        for col in columns:
            if col in filled_df.columns:
                filled_df[col] = filled_df[col].fillna(filled_df[col].mean())
        
        self.df = filled_df
        return self
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        filled_df = self.df.copy()
        for col in columns:
            if col in filled_df.columns:
                filled_df[col] = filled_df[col].fillna(filled_df[col].median())
        
        self.df = filled_df
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'cleaned_rows': self.df.shape[0],
            'removed_rows': self.get_removed_count(),
            'columns': self.df.shape[1],
            'missing_values': self.df.isnull().sum().sum()
        }
        return summary

def example_usage():
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'feature3': np.random.uniform(0, 1, 1000)
    }
    
    df = pd.DataFrame(data)
    df.loc[np.random.choice(df.index, 50), 'feature1'] = np.nan
    
    cleaner = DataCleaner(df)
    cleaner.remove_outliers_iqr(['feature1', 'feature2'])
    cleaner.fill_missing_mean(['feature1'])
    cleaner.normalize_minmax(['feature1', 'feature2', 'feature3'])
    
    cleaned_df = cleaner.get_cleaned_data()
    summary = cleaner.get_summary()
    
    return cleaned_df, summary

if __name__ == "__main__":
    cleaned_data, stats_summary = example_usage()
    print(f"Data cleaning completed")
    print(f"Removed {stats_summary['removed_rows']} outliers")
    print(f"Final dataset shape: {cleaned_data.shape}")
import pandas as pd
import numpy as np
from datetime import datetime

def clean_dataset(input_file, output_file):
    df = pd.read_csv(input_file)
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Standardize date column if exists
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    
    # Fill missing numeric values with column mean
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Remove rows with too many missing values
    threshold = len(df.columns) * 0.3
    df = df.dropna(thresh=threshold)
    
    # Save cleaned data
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    print(f"Original rows: {len(pd.read_csv(input_file))}, Cleaned rows: {len(df)}")

if __name__ == "__main__":
    clean_dataset('raw_data.csv', 'cleaned_data.csv')
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    threshold (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df.copy()

def normalize_column_zscore(dataframe, column):
    """
    Normalize a column using Z-score normalization.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    result_df = dataframe.copy()
    mean_val = result_df[column].mean()
    std_val = result_df[column].std()
    
    if std_val > 0:
        result_df[f'{column}_normalized'] = (result_df[column] - mean_val) / std_val
    else:
        result_df[f'{column}_normalized'] = 0
    
    return result_df

def detect_skewed_columns(dataframe, skew_threshold=0.5):
    """
    Detect columns with significant skewness.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    skew_threshold (float): Absolute skewness threshold
    
    Returns:
    dict: Dictionary with column names and their skewness values
    """
    skewed_columns = {}
    
    for col in dataframe.select_dtypes(include=[np.number]).columns:
        skewness = dataframe[col].skew()
        if abs(skewness) > skew_threshold:
            skewed_columns[col] = skewness
    
    return skewed_columns

def apply_log_transform(dataframe, column):
    """
    Apply log transformation to reduce skewness.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to transform
    
    Returns:
    pd.DataFrame: DataFrame with transformed column
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    result_df = dataframe.copy()
    
    if (result_df[column] <= 0).any():
        min_val = result_df[column].min()
        if min_val <= 0:
            shift = abs(min_val) + 1
            result_df[column] = result_df[column] + shift
    
    result_df[f'{column}_log'] = np.log1p(result_df[column])
    
    return result_df

def clean_dataset(dataframe, numeric_columns=None, outlier_threshold=1.5):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric columns to process
    outlier_threshold (float): IQR threshold for outlier removal
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = dataframe.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_threshold)
            cleaned_df = normalize_column_zscore(cleaned_df, col)
    
    return cleaned_df

def validate_cleaned_data(dataframe, original_dataframe):
    """
    Validate that cleaning operations were successful.
    
    Parameters:
    dataframe (pd.DataFrame): Cleaned DataFrame
    original_dataframe (pd.DataFrame): Original DataFrame
    
    Returns:
    dict: Validation metrics
    """
    validation_results = {
        'rows_removed': len(original_dataframe) - len(dataframe),
        'percentage_removed': (len(original_dataframe) - len(dataframe)) / len(original_dataframe) * 100,
        'original_columns': len(original_dataframe.columns),
        'cleaned_columns': len(dataframe.columns),
        'new_columns_added': len(set(dataframe.columns) - set(original_dataframe.columns))
    }
    
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col in original_dataframe.columns:
            validation_results[f'{col}_original_mean'] = original_dataframe[col].mean()
            validation_results[f'{col}_cleaned_mean'] = dataframe[col].mean()
            validation_results[f'{col}_original_std'] = original_dataframe[col].std()
            validation_results[f'{col}_cleaned_std'] = dataframe[col].std()
    
    return validation_results
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
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            except Exception as e:
                print(f"Warning: Could not clean column '{col}': {e}")
    
    return cleaned_df

def calculate_statistics(df):
    """
    Calculate basic statistics for numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    pd.DataFrame: Statistics summary
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    stats = pd.DataFrame({
        'mean': numeric_df.mean(),
        'median': numeric_df.median(),
        'std': numeric_df.std(),
        'min': numeric_df.min(),
        'max': numeric_df.max(),
        'count': numeric_df.count()
    })
    
    return stats

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    print("Original data shape:", df.shape)
    print("\nOriginal statistics:")
    print(calculate_statistics(df))
    
    cleaned_df = clean_numeric_data(df)
    print("\nCleaned data shape:", cleaned_df.shape)
    print("\nCleaned statistics:")
    print(calculate_statistics(cleaned_df))