
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for duplicates
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

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by converting to appropriate types and handling errors.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to clean
    
    Returns:
        DataFrame with cleaned numeric columns
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of required column names
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"
def remove_duplicates(data_list):
    """
    Remove duplicate entries from a list while preserving order.
    Returns a new list with unique elements.
    """
    seen = set()
    unique_list = []
    for item in data_list:
        if item not in seen:
            seen.add(item)
            unique_list.append(item)
    return unique_list

def clean_numeric_data(values, default=0):
    """
    Clean numeric data by converting strings to floats.
    Non-numeric values are replaced with the default value.
    """
    cleaned = []
    for val in values:
        try:
            cleaned.append(float(val))
        except (ValueError, TypeError):
            cleaned.append(default)
    return cleaned

def filter_by_threshold(data, threshold, keep_above=True):
    """
    Filter data based on a threshold value.
    If keep_above is True, keep values above threshold.
    If keep_above is False, keep values below or equal to threshold.
    """
    if keep_above:
        return [x for x in data if x > threshold]
    else:
        return [x for x in data if x <= threshold]
import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    subset (list, optional): Columns to consider for duplicates
    keep (str): Which duplicates to keep - 'first', 'last', or False
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    if subset is not None:
        if not all(col in df.columns for col in subset):
            raise ValueError("All subset columns must exist in DataFrame")
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep, ignore_index=True)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_numeric_columns(df, columns, fill_method='mean'):
    """
    Clean numeric columns by handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): Numeric columns to clean
    fill_method (str): Method to fill missing values - 'mean', 'median', or 'zero'
    
    Returns:
    pd.DataFrame: DataFrame with cleaned numeric columns
    """
    if df.empty:
        return df
    
    df_cleaned = df.copy()
    
    for col in columns:
        if col not in df_cleaned.columns:
            print(f"Warning: Column '{col}' not found in DataFrame")
            continue
        
        if not pd.api.types.is_numeric_dtype(df_cleaned[col]):
            print(f"Warning: Column '{col}' is not numeric")
            continue
        
        missing_count = df_cleaned[col].isna().sum()
        
        if missing_count > 0:
            if fill_method == 'mean':
                fill_value = df_cleaned[col].mean()
            elif fill_method == 'median':
                fill_value = df_cleaned[col].median()
            elif fill_method == 'zero':
                fill_value = 0
            else:
                raise ValueError("fill_method must be 'mean', 'median', or 'zero'")
            
            df_cleaned[col] = df_cleaned[col].fillna(fill_value)
            print(f"Filled {missing_count} missing values in column '{col}' with {fill_method}")
    
    return df_cleaned

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): Columns that must exist
    min_rows (int): Minimum number of rows required
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns is not None:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    return True

def get_data_summary(df):
    """
    Generate a summary of the DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    dict: Summary statistics
    """
    if df.empty:
        return {"rows": 0, "columns": 0, "empty": True}
    
    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "memory_usage": df.memory_usage(deep=True).sum(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isna().sum().to_dict()
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary["numeric_stats"] = df[numeric_cols].describe().to_dict()
    
    return summary

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', 'David', 'Eve'],
        'score': [85, 90, 90, None, 78, 78, 92],
        'age': [25, 30, 30, 35, None, 28, 32]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    df_cleaned = remove_duplicates(df, subset=['id', 'name'])
    print("After removing duplicates:")
    print(df_cleaned)
    print()
    
    df_filled = clean_numeric_columns(df_cleaned, ['score', 'age'], fill_method='mean')
    print("After cleaning numeric columns:")
    print(df_filled)
    print()
    
    is_valid = validate_dataframe(df_filled, required_columns=['id', 'name', 'score'])
    print(f"DataFrame validation: {is_valid}")
    print()
    
    summary = get_data_summary(df_filled)
    print("DataFrame summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
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
        
        removed_count = self.original_shape[0] - df_clean.shape[0]
        self.df = df_clean.reset_index(drop=True)
        return removed_count
    
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
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_filled = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                median_val = self.df[col].median()
                df_filled[col] = self.df[col].fillna(median_val)
        
        self.df = df_filled
        return self
    
    def get_cleaned_data(self):
        return self.df.copy()
    
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

def example_usage():
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'feature3': np.random.uniform(0, 1, 1000)
    }
    
    df = pd.DataFrame(data)
    df.iloc[10:20, 0] = np.nan
    
    cleaner = DataCleaner(df)
    print(f"Original shape: {df.shape}")
    
    removed = cleaner.remove_outliers_iqr(['feature1', 'feature2'])
    print(f"Removed {removed} outliers")
    
    cleaner.fill_missing_median()
    cleaner.standardize_zscore(['feature1', 'feature2'])
    cleaner.normalize_minmax(['feature3'])
    
    cleaned_df = cleaner.get_cleaned_data()
    summary = cleaner.get_summary()
    
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Summary: {summary}")
    
    return cleaned_df
import pandas as pd
import numpy as np
import sys

def clean_csv(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
        print(f"Original shape: {df.shape}")
        
        df_cleaned = df.copy()
        
        numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df_cleaned[col].isnull().sum() > 0:
                df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
        
        categorical_columns = df_cleaned.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df_cleaned[col].isnull().sum() > 0:
                df_cleaned[col].fillna('Unknown', inplace=True)
        
        df_cleaned = df_cleaned.drop_duplicates()
        
        df_cleaned.to_csv(output_file, index=False)
        print(f"Cleaned shape: {df_cleaned.shape}")
        print(f"Cleaned data saved to: {output_file}")
        
        missing_report = df_cleaned.isnull().sum()
        if missing_report.sum() == 0:
            print("No missing values remaining")
        else:
            print("Remaining missing values:")
            print(missing_report[missing_report > 0])
            
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print("Error: Input file is empty")
        sys.exit(1)
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_cleaner.py <input_file.csv> <output_file.csv>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    clean_csv(input_file, output_file)