
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

def normalize_column(df, column):
    """
    Normalize a column using min-max scaling.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    min_val = df_copy[column].min()
    max_val = df_copy[column].max()
    
    if max_val == min_val:
        df_copy[column] = 0.5
    else:
        df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
    
    return df_copy

def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Imputation strategy ('mean', 'median', 'mode', or 'drop')
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        return df_copy.dropna(subset=numeric_cols)
    
    for col in numeric_cols:
        if df_copy[col].isnull().any():
            if strategy == 'mean':
                fill_value = df_copy[col].mean()
            elif strategy == 'median':
                fill_value = df_copy[col].median()
            elif strategy == 'mode':
                fill_value = df_copy[col].mode()[0]
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            df_copy[col] = df_copy[col].fillna(fill_value)
    
    return df_copy

def clean_dataset(df, numeric_columns=None, outlier_method='iqr', missing_strategy='mean'):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric columns to process
    outlier_method (str): Method for outlier removal
    missing_strategy (str): Strategy for handling missing values
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    df_copy = df.copy()
    
    if numeric_columns is None:
        numeric_columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    
    df_copy = handle_missing_values(df_copy, strategy=missing_strategy)
    
    if outlier_method == 'iqr':
        for col in numeric_columns:
            if col in df_copy.columns:
                df_copy = remove_outliers_iqr(df_copy, col)
    
    for col in numeric_columns:
        if col in df_copy.columns:
            df_copy = normalize_column(df_copy, col)
    
    return df_copy.reset_index(drop=True)
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
                
        self.df = df_clean.reset_index(drop=True)
        return self
        
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
        
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]
        
    def summary(self):
        print(f"Original shape: {self.original_shape}")
        print(f"Cleaned shape: {self.df.shape}")
        print(f"Rows removed: {self.get_removed_count()}")
        print(f"Columns: {list(self.df.columns)}")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print("\nNumeric columns summary:")
            for col in numeric_cols[:5]:
                print(f"  {col}: mean={self.df[col].mean():.3f}, std={self.df[col].std():.3f}")
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a specified column using the IQR method.
    
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
    
    for column in columns:
        if column in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[column]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'id': range(1, 101),
        'value': np.concatenate([
            np.random.normal(100, 10, 90),
            np.random.normal(300, 50, 10)
        ]),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Original data shape: {df.shape}")
    
    cleaned_df = clean_numeric_data(df, columns=['value'])
    print(f"Cleaned data shape: {cleaned_df.shape}")
    
    print("\nSummary statistics:")
    print(f"Original - Mean: {df['value'].mean():.2f}, Std: {df['value'].std():.2f}")
    print(f"Cleaned  - Mean: {cleaned_df['value'].mean():.2f}, Std: {cleaned_df['value'].std():.2f}")import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
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

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistics
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

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing NaN values and converting to appropriate types.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean (default: all numeric columns)
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            cleaned_df = cleaned_df.dropna(subset=[col])
    
    return cleaned_df.reset_index(drop=True)

if __name__ == "__main__":
    sample_data = {
        'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 100, 12, 13, 12, 11, 14]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original data:")
    print(df)
    print(f"\nOriginal shape: {df.shape}")
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("\nData after outlier removal:")
    print(cleaned_df)
    print(f"\nCleaned shape: {cleaned_df.shape}")
    
    stats = calculate_basic_stats(cleaned_df, 'values')
    print("\nBasic statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas Series using the IQR method.
    Returns a cleaned Series.
    """
    if not isinstance(data, pd.Series):
        raise TypeError("Input data must be a pandas Series")
    
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return data[(data >= lower_bound) & (data <= upper_bound)]

def normalize_minmax(data):
    """
    Normalize data to [0, 1] range using min-max scaling.
    Handles both pandas Series and numpy arrays.
    """
    if isinstance(data, pd.Series):
        data_values = data.values
    else:
        data_values = np.array(data)
    
    if len(data_values) == 0:
        return data_values
    
    data_min = np.min(data_values)
    data_max = np.max(data_values)
    
    if data_max == data_min:
        return np.zeros_like(data_values)
    
    normalized = (data_values - data_min) / (data_max - data_min)
    
    if isinstance(data, pd.Series):
        return pd.Series(normalized, index=data.index)
    return normalized

def z_score_normalize(data):
    """
    Normalize data using z-score standardization.
    Returns data with mean=0 and std=1.
    """
    if isinstance(data, pd.Series):
        data_values = data.values
    else:
        data_values = np.array(data)
    
    if len(data_values) < 2:
        return data_values
    
    mean_val = np.mean(data_values)
    std_val = np.std(data_values)
    
    if std_val == 0:
        return np.zeros_like(data_values)
    
    normalized = (data_values - mean_val) / std_val
    
    if isinstance(data, pd.Series):
        return pd.Series(normalized, index=data.index)
    return normalized

def clean_dataset(df, numeric_columns=None):
    """
    Clean a DataFrame by removing outliers and normalizing numeric columns.
    Returns a cleaned DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    cleaned_df = df.copy()
    
    if numeric_columns is None:
        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            # Remove outliers
            cleaned_series = remove_outliers_iqr(cleaned_df[col], col)
            cleaned_df = cleaned_df.loc[cleaned_series.index]
            
            # Normalize the column
            cleaned_df[col] = z_score_normalize(cleaned_df[col])
    
    return cleaned_df.reset_index(drop=True)

def validate_data(data, check_nan=True, check_inf=True):
    """
    Validate data for common issues like NaN values and infinities.
    Returns a dictionary with validation results.
    """
    validation_result = {
        'has_nan': False,
        'has_inf': False,
        'is_empty': False,
        'sample_count': 0
    }
    
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        data_values = data.values
    else:
        data_values = np.array(data)
    
    if len(data_values) == 0:
        validation_result['is_empty'] = True
        return validation_result
    
    validation_result['sample_count'] = len(data_values)
    
    if check_nan:
        validation_result['has_nan'] = np.any(np.isnan(data_values))
    
    if check_inf:
        validation_result['has_inf'] = np.any(np.isinf(data_values))
    
    return validation_result
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
    
    return filtered_df

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name
    
    Returns:
    dict: Dictionary containing summary statistics
    """
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

def normalize_column(df, column):
    """
    Normalize a column using min-max scaling.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = df[column].min()
    max_val = df[column].max()
    
    if max_val == min_val:
        df[f'{column}_normalized'] = 0.5
    else:
        df[f'{column}_normalized'] = (df[column] - min_val) / (max_val - min_val)
    
    return df

def main():
    """
    Example usage of data cleaning functions.
    """
    np.random.seed(42)
    sample_data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100)
    }
    
    df = pd.DataFrame(sample_data)
    
    print("Original DataFrame shape:", df.shape)
    print("Original summary statistics:")
    print(calculate_summary_stats(df, 'value'))
    
    cleaned_df = remove_outliers_iqr(df, 'value')
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned summary statistics:")
    print(calculate_summary_stats(cleaned_df, 'value'))
    
    normalized_df = normalize_column(cleaned_df, 'value')
    print("\nNormalized column added:")
    print(normalized_df[['value', 'value_normalized']].head())

if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd

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

def clean_dataset(df, columns_to_clean=None):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary of statistics for each cleaned column
    """
    if columns_to_clean is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns_to_clean = list(numeric_cols)
    
    cleaned_df = df.copy()
    statistics = {}
    
    for column in columns_to_clean:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            
            stats = calculate_summary_statistics(cleaned_df, column)
            stats['outliers_removed'] = removed_count
            statistics[column] = stats
    
    return cleaned_df, statistics

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[::100, 'A'] = 500
    
    print("Original dataset shape:", df.shape)
    print("\nOriginal statistics for column 'A':")
    print(calculate_summary_statistics(df, 'A'))
    
    cleaned_df, stats = clean_dataset(df, ['A', 'B'])
    
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("\nCleaned statistics for column 'A':")
    print(stats['A'])