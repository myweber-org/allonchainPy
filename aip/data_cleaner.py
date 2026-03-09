
def remove_duplicates(data_list):
    """
    Remove duplicate entries from a list while preserving order.
    
    Args:
        data_list: A list of elements (must be hashable)
    
    Returns:
        A new list with duplicates removed
    """
    seen = set()
    result = []
    
    for item in data_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    
    return result

def clean_data_with_key(data_list, key_func=None):
    """
    Remove duplicates based on a key function.
    
    Args:
        data_list: A list of elements
        key_func: Function to extract comparison key (optional)
    
    Returns:
        A new list with duplicates removed based on key
    """
    if key_func is None:
        return remove_duplicates(data_list)
    
    seen = set()
    result = []
    
    for item in data_list:
        key = key_func(item)
        if key not in seen:
            seen.add(key)
            result.append(item)
    
    return result

if __name__ == "__main__":
    # Example usage
    sample_data = [1, 2, 2, 3, 4, 4, 5]
    cleaned = remove_duplicates(sample_data)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {cleaned}")
    
    # Example with custom key
    data_with_dicts = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
        {"id": 1, "name": "Alice"},
        {"id": 3, "name": "Charlie"}
    ]
    
    cleaned_dicts = clean_data_with_key(data_with_dicts, key_func=lambda x: x["id"])
    print(f"\nOriginal dicts: {data_with_dicts}")
    print(f"Cleaned dicts: {cleaned_dicts}")import pandas as pd

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
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows.")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): Columns that must be present.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame.")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty.")
        return True
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    return True

def clean_numeric_column(df, column_name, fill_method='mean'):
    """
    Clean numeric column by handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Name of column to clean.
        fill_method (str): Method to fill missing values ('mean', 'median', 'zero').
    
    Returns:
        pd.DataFrame: DataFrame with cleaned column.
    """
    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' not found in DataFrame.")
        return df
    
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        print(f"Error: Column '{column_name}' is not numeric.")
        return df
    
    missing_count = df[column_name].isna().sum()
    
    if missing_count > 0:
        if fill_method == 'mean':
            fill_value = df[column_name].mean()
        elif fill_method == 'median':
            fill_value = df[column_name].median()
        elif fill_method == 'zero':
            fill_value = 0
        else:
            print(f"Warning: Unknown fill method '{fill_method}', using mean.")
            fill_value = df[column_name].mean()
        
        df[column_name] = df[column_name].fillna(fill_value)
        print(f"Filled {missing_count} missing values in '{column_name}' with {fill_method}.")
    
    return df

def process_dataframe(df, operations=None):
    """
    Apply multiple cleaning operations to a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        operations (list, optional): List of cleaning operations to apply.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    if not validate_dataframe(df):
        return df
    
    result_df = df.copy()
    
    if operations:
        for operation in operations:
            if operation['type'] == 'remove_duplicates':
                result_df = remove_duplicates(
                    result_df, 
                    subset=operation.get('subset'), 
                    keep=operation.get('keep', 'first')
                )
            elif operation['type'] == 'clean_numeric':
                result_df = clean_numeric_column(
                    result_df,
                    column_name=operation['column'],
                    fill_method=operation.get('fill_method', 'mean')
                )
    
    return result_df
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    outliers_removed = len(data) - len(filtered_data)
    
    return filtered_data, outliers_removed

def z_score_normalize(data, column):
    """
    Normalize data using z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column]
    
    normalized = (data[column] - mean_val) / std_val
    return normalized

def min_max_normalize(data, column, feature_range=(0, 1)):
    """
    Normalize data using min-max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column]
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    
    if feature_range != (0, 1):
        min_target, max_target = feature_range
        normalized = normalized * (max_target - min_target) + min_target
    
    return normalized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns
    """
    if columns is None:
        columns = data.columns
    
    data_copy = data.copy()
    
    for col in columns:
        if col not in data_copy.columns:
            continue
            
        missing_count = data_copy[col].isnull().sum()
        if missing_count == 0:
            continue
            
        if strategy == 'mean':
            fill_value = data_copy[col].mean()
        elif strategy == 'median':
            fill_value = data_copy[col].median()
        elif strategy == 'mode':
            fill_value = data_copy[col].mode()[0]
        elif strategy == 'drop':
            data_copy = data_copy.dropna(subset=[col])
            continue
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        data_copy[col] = data_copy[col].fillna(fill_value)
    
    return data_copy

def detect_skewed_columns(data, threshold=0.5):
    """
    Detect columns with skewed distributions
    """
    skewed_columns = []
    
    for col in data.select_dtypes(include=[np.number]).columns:
        skewness = stats.skew(data[col].dropna())
        if abs(skewness) > threshold:
            skewed_columns.append((col, skewness))
    
    return sorted(skewed_columns, key=lambda x: abs(x[1]), reverse=True)
import pandas as pd
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
    
    return filtered_df

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

def clean_dataset(df, numeric_columns):
    """
    Clean dataset by removing outliers from multiple numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        numeric_columns (list): List of numeric column names to clean.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[column]):
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[::100, 'A'] = 500
    
    print("Original dataset shape:", df.shape)
    print("Original statistics for column 'A':")
    print(calculate_summary_statistics(df, 'A'))
    
    cleaned_df = clean_dataset(df, ['A', 'B', 'C'])
    
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("Cleaned statistics for column 'A':")
    print(calculate_summary_statistics(cleaned_df, 'A'))
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        multiplier: IQR multiplier for outlier detection
        
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
        data[column + '_normalized'] = 0.5
    else:
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
        data[column + '_standardized'] = 0
    else:
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

def create_sample_data():
    """
    Create sample data for testing the cleaning functions.
    
    Returns:
        Sample pandas DataFrame
    """
    np.random.seed(42)
    data = {
        'id': range(1, 101),
        'value': np.random.normal(100, 15, 100),
        'score': np.random.uniform(0, 100, 100)
    }
    
    df = pd.DataFrame(data)
    
    df.loc[10:15, 'value'] = np.nan
    df.loc[95, 'value'] = 500
    df.loc[96, 'value'] = -200
    
    return df

if __name__ == "__main__":
    sample_data = create_sample_data()
    print("Original data shape:", sample_data.shape)
    print("Missing values:", sample_data['value'].isnull().sum())
    
    cleaned_data = handle_missing_values(sample_data, strategy='mean')
    cleaned_data = remove_outliers_iqr(cleaned_data, 'value')
    cleaned_data = normalize_minmax(cleaned_data, 'score')
    cleaned_data = standardize_zscore(cleaned_data, 'value')
    
    print("Cleaned data shape:", cleaned_data.shape)
    print("Data cleaning completed successfully.")
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
            print(f"Loaded data with shape: {self.df.shape}")
            return True
        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
            return False
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def remove_duplicates(self):
        if self.df is not None:
            initial_rows = len(self.df)
            self.df.drop_duplicates(inplace=True)
            removed = initial_rows - len(self.df)
            print(f"Removed {removed} duplicate rows")
            return removed
        return 0
    
    def fill_missing_values(self, strategy='mean', fill_value=None):
        if self.df is None:
            print("No data loaded")
            return
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_val = self.df[col].mean()
                elif strategy == 'median':
                    fill_val = self.df[col].median()
                elif strategy == 'mode':
                    fill_val = self.df[col].mode()[0]
                elif strategy == 'custom' and fill_value is not None:
                    fill_val = fill_value
                else:
                    continue
                
                self.df[col].fillna(fill_val, inplace=True)
                print(f"Filled missing values in column '{col}' with {fill_val}")
    
    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if self.df is None:
            print("No data loaded")
            return 0
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        initial_rows = len(self.df)
        
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        
        removed = initial_rows - len(self.df)
        print(f"Removed {removed} outliers using IQR method")
        return removed
    
    def standardize_columns(self, columns=None):
        if self.df is None:
            print("No data loaded")
            return
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                mean = self.df[col].mean()
                std = self.df[col].std()
                if std > 0:
                    self.df[col] = (self.df[col] - mean) / std
                    print(f"Standardized column '{col}'")
    
    def save_cleaned_data(self, output_path=None):
        if self.df is None:
            print("No data to save")
            return False
        
        if output_path is None:
            output_path = self.file_path.parent / f"cleaned_{self.file_path.name}"
        
        try:
            self.df.to_csv(output_path, index=False)
            print(f"Saved cleaned data to: {output_path}")
            return True
        except Exception as e:
            print(f"Error saving file: {e}")
            return False
    
    def get_summary(self):
        if self.df is None:
            return "No data loaded"
        
        summary = {
            'shape': self.df.shape,
            'missing_values': self.df.isnull().sum().sum(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(exclude=[np.number]).columns)
        }
        
        return summary

def clean_csv_file(input_file, output_file=None):
    cleaner = DataCleaner(input_file)
    
    if not cleaner.load_data():
        return None
    
    print("Starting data cleaning process...")
    
    cleaner.remove_duplicates()
    cleaner.fill_missing_values(strategy='median')
    cleaner.remove_outliers_iqr(multiplier=1.5)
    cleaner.standardize_columns()
    
    if output_file:
        cleaner.save_cleaned_data(output_file)
    else:
        cleaner.save_cleaned_data()
    
    summary = cleaner.get_summary()
    print(f"Cleaning complete. Final shape: {summary['shape']}")
    
    return cleaner.df

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'value': [10.5, 20.3, 15.7, None, 25.1, 1000.0, 18.9, 22.4, None, 19.8],
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_file = "test_data.csv"
    test_df.to_csv(test_file, index=False)
    
    cleaned_df = clean_csv_file(test_file, "cleaned_test_data.csv")
    
    import os
    if os.path.exists(test_file):
        os.remove(test_file)
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
    Clean numeric columns by removing outliers from specified columns.
    If columns is None, clean all numeric columns.
    
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

def validate_dataframe(df):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
    }
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.randint(1, 100, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[0, 'A'] = 1000
    df.loc[1, 'B'] = 500
    
    print("Original DataFrame shape:", df.shape)
    print("\nValidation results:")
    print(validate_dataframe(df))
    
    cleaned_df = clean_numeric_data(df, ['A', 'B'])
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
import pandas as pd
import numpy as np

def clean_csv_data(file_path, fill_method='mean', output_path=None):
    """
    Load a CSV file, clean missing values, and optionally save cleaned data.
    
    Args:
        file_path (str): Path to input CSV file
        fill_method (str): Method for filling missing values ('mean', 'median', 'mode', 'zero')
        output_path (str, optional): Path to save cleaned CSV file
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data with shape: {df.shape}")
        
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if fill_method == 'mean':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif fill_method == 'median':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            elif fill_method == 'mode':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mode().iloc[0])
            elif fill_method == 'zero':
                df[numeric_cols] = df[numeric_cols].fillna(0)
            else:
                raise ValueError(f"Unknown fill method: {fill_method}")
            
            print(f"Missing values filled using '{fill_method}' method")
        
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

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Args:
        df (pandas.DataFrame): Input DataFrame
        column (str): Column name to process
        multiplier (float): IQR multiplier for outlier detection
    
    Returns:
        pandas.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        print(f"Column '{column}' not found in DataFrame")
        return df
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    original_count = len(df)
    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    removed_count = original_count - len(df_clean)
    
    print(f"Removed {removed_count} outliers from column '{column}'")
    
    return df_clean

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, np.nan, 30, 40, 50],
        'C': [100, 200, 300, 400, 500]
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', fill_method='mean', output_path='cleaned_data.csv')
    
    if cleaned_df is not None:
        print("Data cleaning completed successfully")
        print(cleaned_df)