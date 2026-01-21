
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        threshold: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data
    
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        DataFrame with standardized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data
    
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values in numeric columns.
    
    Args:
        data: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        DataFrame with handled missing values
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        return data.dropna(subset=numeric_cols)
    
    for col in numeric_cols:
        if data[col].isnull().any():
            if strategy == 'mean':
                fill_value = data[col].mean()
            elif strategy == 'median':
                fill_value = data[col].median()
            elif strategy == 'mode':
                fill_value = data[col].mode()[0]
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            data[col] = data[col].fillna(fill_value)
    
    return data

def clean_dataset(data, numeric_columns=None, outlier_threshold=1.5, 
                  normalize=True, standardize=False, missing_strategy='mean'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process (default: all numeric)
        outlier_threshold: IQR threshold for outlier removal
        normalize: whether to apply min-max normalization
        standardize: whether to apply z-score standardization
        missing_strategy: strategy for handling missing values
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_data = data.copy()
    
    if numeric_columns is None:
        numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = handle_missing_values(cleaned_data, strategy=missing_strategy)
    
    for col in numeric_columns:
        if col in cleaned_data.columns:
            cleaned_data = remove_outliers_iqr(cleaned_data, col, outlier_threshold)
            
            if normalize:
                cleaned_data = normalize_minmax(cleaned_data, col)
            
            if standardize:
                cleaned_data = standardize_zscore(cleaned_data, col)
    
    return cleaned_dataimport numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Args:
        data: pandas DataFrame
        column: Column name to process
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def normalize_minmax(data, column):
    """
    Normalize column values to range [0, 1] using min-max scaling.
    
    Args:
        data: pandas DataFrame
        column: Column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def z_score_normalize(data, column):
    """
    Normalize column values using z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: Column name to normalize
    
    Returns:
        Series with z-score normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    z_scores = (data[column] - mean_val) / std_val
    return z_scores

def clean_dataset(data, numeric_columns=None, outlier_multiplier=1.5, normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: Input DataFrame
        numeric_columns: List of numeric columns to process (default: all numeric)
        outlier_multiplier: Multiplier for IQR outlier detection
        normalize_method: 'minmax' or 'zscore' normalization
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        cleaned_data = remove_outliers_iqr(cleaned_data, column, outlier_multiplier)
        
        if normalize_method == 'minmax':
            cleaned_data[f'{column}_normalized'] = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data[f'{column}_normalized'] = z_score_normalize(cleaned_data, column)
        else:
            raise ValueError("normalize_method must be 'minmax' or 'zscore'")
    
    return cleaned_data

def validate_data(data, required_columns, allow_nan=False):
    """
    Validate DataFrame structure and content.
    
    Args:
        data: DataFrame to validate
        required_columns: List of required column names
        allow_nan: Whether NaN values are allowed
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"
    
    if not allow_nan:
        nan_columns = data.columns[data.isnull().any()].tolist()
        if nan_columns:
            return False, f"NaN values found in columns: {nan_columns}"
    
    return True, "Data validation passed"

def example_usage():
    """Example demonstrating the data cleaning functions."""
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'id': range(100),
        'feature_a': np.random.normal(50, 15, 100),
        'feature_b': np.random.exponential(10, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    print("Original data shape:", sample_data.shape)
    print("Original statistics:")
    print(sample_data[['feature_a', 'feature_b']].describe())
    
    cleaned = clean_dataset(
        sample_data,
        numeric_columns=['feature_a', 'feature_b'],
        outlier_multiplier=1.5,
        normalize_method='minmax'
    )
    
    print("\nCleaned data shape:", cleaned.shape)
    print("Cleaned statistics:")
    print(cleaned[['feature_a', 'feature_b']].describe())
    
    is_valid, message = validate_data(
        cleaned,
        required_columns=['feature_a', 'feature_b', 'feature_a_normalized'],
        allow_nan=False
    )
    
    print(f"\nValidation: {message}")
    
    return cleaned

if __name__ == "__main__":
    result = example_usage()
    print(f"\nProcessing complete. Final dataset has {len(result)} rows.")import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list, optional): Column labels to consider for duplicates.
        keep (str, optional): Which duplicates to keep.
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    return cleaned_df

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by converting to numeric and filling NaN with mean.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of column names to clean.
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns.
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
    return df

def validate_dataframe(df, required_columns):
    """
    Validate DataFrame contains required columns.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        bool: True if all required columns are present.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return False
    return Trueimport pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range (IQR) method.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to process.
    
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
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to analyze.
    
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
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to process.
    
    Returns:
        tuple: (cleaned_df, original_stats, cleaned_stats)
    """
    original_stats = calculate_summary_statistics(df, column)
    cleaned_df = remove_outliers_iqr(df, column)
    cleaned_stats = calculate_summary_statistics(cleaned_df, column)
    
    print(f"Original data shape: {df.shape}")
    print(f"Cleaned data shape: {cleaned_df.shape}")
    print(f"Outliers removed: {len(df) - len(cleaned_df)}")
    
    return cleaned_df, original_stats, cleaned_stats
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df: Input pandas DataFrame
        column_mapping: Dictionary to rename columns (old_name: new_name)
        drop_duplicates: Whether to remove duplicate rows
        normalize_text: Whether to normalize text columns (strip, lowercase)
    
    Returns:
        Cleaned pandas DataFrame
    """
    df_clean = df.copy()
    
    if column_mapping:
        df_clean = df_clean.rename(columns=column_mapping)
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    
    if normalize_text:
        for col in df_clean.select_dtypes(include=['object']).columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
    
    return df_clean

def remove_special_characters(text, keep_pattern=r'[a-zA-Z0-9\s]'):
    """
    Remove special characters from text, keeping only specified pattern.
    
    Args:
        text: Input string
        keep_pattern: Regex pattern of characters to keep
    
    Returns:
        Cleaned string
    """
    if pd.isna(text):
        return text
    
    text = str(text)
    return re.sub(f'[^{keep_pattern}]', '', text)

def validate_email(email):
    """
    Validate email format using regex pattern.
    
    Args:
        email: Email string to validate
    
    Returns:
        Boolean indicating if email is valid
    """
    if pd.isna(email):
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email)))import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for identifying duplicates
        keep: determines which duplicates to keep ('first', 'last', False)
    
    Returns:
        DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_numeric_column(df, column_name):
    """
    Clean a numeric column by converting to float and removing NaN values.
    
    Args:
        df: pandas DataFrame
        column_name: name of the column to clean
    
    Returns:
        DataFrame with cleaned column
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    original_count = len(df)
    df = df.dropna(subset=[column_name])
    
    removed_count = original_count - len(df)
    if removed_count > 0:
        print(f"Removed {removed_count} rows with invalid values in column '{column_name}'")
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame
        required_columns: list of column names that must be present
    
    Returns:
        Boolean indicating if validation passed
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    return True

def get_data_summary(df):
    """
    Generate a summary of the DataFrame.
    
    Args:
        df: pandas DataFrame
    
    Returns:
        Dictionary containing summary statistics
    """
    summary = {
        'rows': len(df),
        'columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'column_types': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum()
    }
    
    return summary
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
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
    Clean numeric data by removing outliers from multiple columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{col}'")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
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
    print("Original value statistics:")
    print(df['value'].describe())
    
    if validate_dataframe(df, ['id', 'value']):
        cleaned_df = clean_numeric_data(df, ['value'])
        print("\nCleaned DataFrame shape:", cleaned_df.shape)
        print("Cleaned value statistics:")
        print(cleaned_df['value'].describe())
import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.df.columns
        
        for col in columns:
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
                    fill_value = strategy
                
                self.df[col] = self.df[col].fillna(fill_value)
        return self.df
    
    def remove_outliers_iqr(self, columns=None, threshold=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        
        return self.df
    
    def standardize_columns(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            mean = self.df[col].mean()
            std = self.df[col].std()
            if std > 0:
                self.df[col] = (self.df[col] - mean) / std
        
        return self.df
    
    def get_cleaned_data(self):
        return self.df.copy()

def example_usage():
    data = {
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, 20, 30, np.nan, 50, 60],
        'C': ['x', 'y', 'z', 'x', 'y', 'z']
    }
    
    df = pd.DataFrame(data)
    cleaner = DataCleaner(df)
    
    print("Original data:")
    print(df)
    
    cleaner.handle_missing_values(strategy='mean')
    cleaner.remove_outliers_iqr(threshold=1.5)
    
    print("\nCleaned data:")
    print(cleaner.get_cleaned_data())

if __name__ == "__main__":
    example_usage()