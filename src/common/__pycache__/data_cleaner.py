
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
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        removed_count = self.original_shape[0] - df_clean.shape[0]
        self.df = df_clean
        return removed_count
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_normalized = self.df.copy()
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    df_normalized[col] = (self.df[col] - min_val) / (max_val - min_val)
        
        self.df = df_normalized
        return self.df
    
    def standardize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_standardized = self.df.copy()
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    df_standardized[col] = (self.df[col] - mean_val) / std_val
        
        self.df = df_standardized
        return self.df
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_filled = self.df.copy()
        for col in columns:
            if col in self.df.columns and self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0]
                elif strategy == 'zero':
                    fill_value = 0
                else:
                    continue
                
                df_filled[col] = self.df[col].fillna(fill_value)
        
        self.df = df_filled
        return self.df
    
    def get_cleaned_data(self):
        return self.df.copy()
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'current_rows': self.df.shape[0],
            'columns': self.df.shape[1],
            'removed_rows': self.original_shape[0] - self.df.shape[0],
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns)
        }
        return summary

def example_usage():
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'feature3': np.random.randint(1, 100, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    df.loc[np.random.choice(df.index, 50), 'feature1'] = np.nan
    
    cleaner = DataCleaner(df)
    print(f"Original shape: {cleaner.get_summary()['original_rows']} rows")
    
    removed = cleaner.remove_outliers_iqr(['feature1', 'feature2'])
    print(f"Removed {removed} outliers")
    
    cleaner.handle_missing_values(strategy='mean')
    cleaner.normalize_minmax(['feature1', 'feature2', 'feature3'])
    
    cleaned_df = cleaner.get_cleaned_data()
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Summary: {cleaner.get_summary()}")

if __name__ == "__main__":
    example_usage()
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Clean dataset by removing outliers and normalizing numeric columns
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column not in data.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, column)
        
        if normalize_method == 'minmax':
            cleaned_data[f"{column}_normalized"] = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data[f"{column}_standardized"] = normalize_zscore(cleaned_data, column)
    
    return cleaned_data

def get_data_summary(data):
    """
    Generate summary statistics for the dataset
    """
    summary = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'numeric_columns': data.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': data.select_dtypes(include=['object']).columns.tolist(),
        'missing_values': data.isnull().sum().to_dict(),
        'basic_stats': data.describe().to_dict()
    }
    return summaryimport pandas as pd

def clean_dataset(df, remove_duplicates=True, fill_method='drop'):
    """
    Clean a pandas DataFrame by handling missing values and optionally removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    remove_duplicates (bool): If True, remove duplicate rows.
    fill_method (str): Method to handle missing values. 
                       Options: 'drop' (drop rows), 'fill_mean' (fill with column mean for numeric columns),
                       'fill_median' (fill with column median for numeric columns).
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_method == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_method == 'fill_mean':
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
    elif fill_method == 'fill_median':
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
    else:
        raise ValueError(f"Unsupported fill_method: {fill_method}")
    
    # Remove duplicates if requested
    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    return cleaned_df

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate a DataFrame for basic integrity checks.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (is_valid, message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Dataset is valid"

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = {
#         'A': [1, 2, None, 4, 4],
#         'B': [5, None, 7, 8, 8],
#         'C': ['x', 'y', 'z', 'x', 'x']
#     }
#     df = pd.DataFrame(sample_data)
#     print("Original DataFrame:")
#     print(df)
#     
#     cleaned = clean_dataset(df, remove_duplicates=True, fill_method='fill_mean')
#     print("\nCleaned DataFrame:")
#     print(cleaned)
#     
#     is_valid, message = validate_dataset(cleaned, required_columns=['A', 'B'], min_rows=2)
#     print(f"\nValidation: {is_valid}, Message: {message}")import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
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

def validate_dataframe(df, required_columns=None):
    """
    Validate that the DataFrame meets basic requirements.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 30, 40, 50],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B'])
    print(f"\nValidation: {message}")
import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    for column in df.select_dtypes(include=[np.number]).columns:
        df[column] = df[column].fillna(df[column].median())
    
    # Remove outliers using z-score
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    df = df[(z_scores < 3).all(axis=1)]
    
    # Normalize numeric columns
    for column in numeric_cols:
        if df[column].std() != 0:
            df[column] = (df[column] - df[column].mean()) / df[column].std()
    
    return df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = load_and_clean_data(input_file)
    save_cleaned_data(cleaned_df, output_file)
import pandas as pd
import numpy as np

def clean_csv_data(file_path, fill_method='mean', drop_threshold=0.5):
    """
    Clean CSV data by handling missing values and removing invalid rows.
    
    Parameters:
    file_path (str): Path to the CSV file
    fill_method (str): Method for filling missing values ('mean', 'median', 'mode', 'zero')
    drop_threshold (float): Threshold for dropping rows with too many missing values (0 to 1)
    
    Returns:
    pandas.DataFrame: Cleaned dataframe
    """
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    original_shape = df.shape
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Drop rows with too many missing values
    missing_ratio = df.isnull().sum(axis=1) / df.shape[1]
    df = df[missing_ratio <= drop_threshold]
    
    # Fill missing values based on specified method
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if fill_method == 'mean':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())
    elif fill_method == 'median':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
    elif fill_method == 'zero':
        df[numeric_cols] = df[numeric_cols].fillna(0)
    elif fill_method == 'mode':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
    
    # Fill remaining non-numeric columns with forward fill
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Remove outliers using IQR method for numeric columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    cleaned_shape = df.shape
    
    print(f"Original data shape: {original_shape}")
    print(f"Cleaned data shape: {cleaned_shape}")
    print(f"Rows removed: {original_shape[0] - cleaned_shape[0]}")
    print(f"Columns: {cleaned_shape[1]}")
    
    return df

def save_cleaned_data(df, output_path):
    """
    Save cleaned dataframe to CSV file.
    
    Parameters:
    df (pandas.DataFrame): Cleaned dataframe
    output_path (str): Path to save the cleaned CSV file
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    try:
        cleaned_df = clean_csv_data(input_file, fill_method='median', drop_threshold=0.3)
        save_cleaned_data(cleaned_df, output_file)
    except Exception as e:
        print(f"Error during data cleaning: {e}")
def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed from the specified column.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data
import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

def remove_outliers(data, column, threshold=1.5):
    """
    Remove outliers from specified column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, columns=None):
    """
    Normalize data using Min-Max scaling
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    normalized_data = data.copy()
    for col in columns:
        if col in data.columns and np.issubdtype(data[col].dtype, np.number):
            min_val = data[col].min()
            max_val = data[col].max()
            if max_val != min_val:
                normalized_data[col] = (data[col] - min_val) / (max_val - min_val)
    
    return normalized_data

def normalize_zscore(data, columns=None):
    """
    Normalize data using Z-score standardization
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    standardized_data = data.copy()
    for col in columns:
        if col in data.columns and np.issubdtype(data[col].dtype, np.number):
            mean_val = data[col].mean()
            std_val = data[col].std()
            if std_val != 0:
                standardized_data[col] = (data[col] - mean_val) / std_val
    
    return standardized_data

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values with specified strategy
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    cleaned_data = data.copy()
    
    for col in columns:
        if col not in data.columns:
            continue
            
        if data[col].isnull().any():
            if strategy == 'mean':
                fill_value = data[col].mean()
            elif strategy == 'median':
                fill_value = data[col].median()
            elif strategy == 'mode':
                fill_value = data[col].mode()[0] if not data[col].mode().empty else 0
            elif strategy == 'drop':
                cleaned_data = cleaned_data.dropna(subset=[col])
                continue
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            cleaned_data[col] = cleaned_data[col].fillna(fill_value)
    
    return cleaned_data

def get_data_summary(data):
    """
    Generate comprehensive data summary
    """
    summary = {
        'shape': data.shape,
        'dtypes': data.dtypes.to_dict(),
        'missing_values': data.isnull().sum().to_dict(),
        'numeric_stats': {}
    }
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['numeric_stats'][col] = {
            'mean': data[col].mean(),
            'std': data[col].std(),
            'min': data[col].min(),
            '25%': data[col].quantile(0.25),
            '50%': data[col].median(),
            '75%': data[col].quantile(0.75),
            'max': data[col].max(),
            'skewness': data[col].skew(),
            'kurtosis': data[col].kurtosis()
        }
    
    return summary
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
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        removed_count = self.original_shape[0] - df_clean.shape[0]
        self.df = df_clean
        return removed_count
    
    def normalize_data(self, columns=None, method='zscore'):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_normalized = self.df.copy()
        for col in columns:
            if col in df_normalized.columns:
                if method == 'zscore':
                    df_normalized[col] = stats.zscore(df_normalized[col])
                elif method == 'minmax':
                    df_normalized[col] = (df_normalized[col] - df_normalized[col].min()) / (df_normalized[col].max() - df_normalized[col].min())
                elif method == 'robust':
                    median = df_normalized[col].median()
                    iqr = df_normalized[col].quantile(0.75) - df_normalized[col].quantile(0.25)
                    df_normalized[col] = (df_normalized[col] - median) / iqr
        
        self.df = df_normalized
        return self.df
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_filled = self.df.copy()
        for col in columns:
            if col in df_filled.columns and df_filled[col].isnull().any():
                if strategy == 'mean':
                    fill_value = df_filled[col].mean()
                elif strategy == 'median':
                    fill_value = df_filled[col].median()
                elif strategy == 'mode':
                    fill_value = df_filled[col].mode()[0]
                elif strategy == 'drop':
                    df_filled = df_filled.dropna(subset=[col])
                    continue
                else:
                    fill_value = 0
                
                df_filled[col] = df_filled[col].fillna(fill_value)
        
        self.df = df_filled
        return self.df
    
    def get_clean_data(self):
        return self.df
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'current_rows': self.df.shape[0],
            'columns': self.df.shape[1],
            'missing_values': self.df.isnull().sum().sum(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns)
        }
        return summary

def example_usage():
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'feature3': np.random.randint(1, 100, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    df.loc[np.random.choice(df.index, 50), 'feature1'] = np.nan
    df.loc[np.random.choice(df.index, 20), 'feature2'] = 1000
    
    cleaner = DataCleaner(df)
    print("Initial summary:", cleaner.get_summary())
    
    removed = cleaner.remove_outliers_iqr(['feature1', 'feature2'])
    print(f"Removed {removed} outliers")
    
    cleaner.handle_missing_values(strategy='mean')
    cleaner.normalize_data(method='zscore')
    
    print("Final summary:", cleaner.get_summary())
    return cleaner.get_clean_data()

if __name__ == "__main__":
    cleaned_data = example_usage()
    print("Data cleaning completed successfully")
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def standardize_zscore(data, column):
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
            cleaned_df = standardize_zscore(cleaned_df, col)
    return cleaned_df

def summary_statistics(df, column):
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    return stats

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    numeric_cols = ['feature1', 'feature2']
    result = clean_dataset(sample_data, numeric_cols)
    
    print("Original shape:", sample_data.shape)
    print("Cleaned shape:", result.shape)
    print("\nFeature1 statistics:")
    print(summary_statistics(result, 'feature1'))import numpy as np
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
    
    def normalize_column(self, column, method='minmax'):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise ValueError(f"Column '{column}' is not numeric")
        
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            if max_val != min_val:
                self.df[f"{column}_normalized"] = (self.df[column] - min_val) / (max_val - min_val)
            else:
                self.df[f"{column}_normalized"] = 0.5
        
        elif method == 'zscore':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            if std_val > 0:
                self.df[f"{column}_normalized"] = (self.df[column] - mean_val) / std_val
            else:
                self.df[f"{column}_normalized"] = 0
        
        else:
            raise ValueError("Method must be 'minmax' or 'zscore'")
        
        return self.df[f"{column}_normalized"]
    
    def fill_missing(self, strategy='mean', custom_value=None):
        df_filled = self.df.copy()
        
        for col in df_filled.columns:
            if df_filled[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df_filled[col]):
                    if strategy == 'mean':
                        fill_value = df_filled[col].mean()
                    elif strategy == 'median':
                        fill_value = df_filled[col].median()
                    elif strategy == 'custom' and custom_value is not None:
                        fill_value = custom_value
                    else:
                        continue
                    
                    df_filled[col] = df_filled[col].fillna(fill_value)
                else:
                    if strategy == 'mode':
                        fill_value = df_filled[col].mode()[0] if not df_filled[col].mode().empty else 'Unknown'
                    elif strategy == 'custom' and custom_value is not None:
                        fill_value = custom_value
                    else:
                        continue
                    
                    df_filled[col] = df_filled[col].fillna(fill_value)
        
        self.df = df_filled
        return self.df
    
    def get_cleaned_data(self):
        return self.df.copy()
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'current_rows': self.df.shape[0],
            'columns': self.df.shape[1],
            'missing_values': self.df.isnull().sum().sum(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object', 'category']).columns)
        }
        return summary

def example_usage():
    np.random.seed(42)
    data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'score': np.random.uniform(0, 1, 100)
    }
    
    df = pd.DataFrame(data)
    df.loc[10:15, 'value'] = np.nan
    df.loc[95, 'value'] = 500
    
    cleaner = DataCleaner(df)
    print("Initial summary:", cleaner.get_summary())
    
    removed = cleaner.remove_outliers_iqr(['value'])
    print(f"Removed {removed} outliers")
    
    cleaner.fill_missing(strategy='mean')
    cleaner.normalize_column('value', method='minmax')
    
    print("Final summary:", cleaner.get_summary())
    return cleaner.get_cleaned_data()

if __name__ == "__main__":
    cleaned_df = example_usage()
    print("\nFirst 5 rows of cleaned data:")
    print(cleaned_df.head())import pandas as pd
import numpy as np

def clean_dataset(df, strategy='mean', outlier_threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'drop').
    outlier_threshold (float): Number of standard deviations to consider as outlier.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    else:
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            if cleaned_df[column].isnull().any():
                if strategy == 'mean':
                    fill_value = cleaned_df[column].mean()
                elif strategy == 'median':
                    fill_value = cleaned_df[column].median()
                elif strategy == 'mode':
                    fill_value = cleaned_df[column].mode()[0]
                else:
                    fill_value = 0
                cleaned_df[column].fillna(fill_value, inplace=True)
    
    # Remove outliers using Z-score method
    numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs((cleaned_df[numeric_columns] - cleaned_df[numeric_columns].mean()) / 
                      cleaned_df[numeric_columns].std())
    outlier_mask = (z_scores < outlier_threshold).all(axis=1)
    cleaned_df = cleaned_df[outlier_mask]
    
    # Reset index after cleaning
    cleaned_df.reset_index(drop=True, inplace=True)
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "Data validation passed"

# Example usage
if __name__ == "__main__":
    # Create sample data with missing values and outliers
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],  # 100 is an outlier
        'B': [5, 6, 7, np.nan, 8],
        'C': ['x', 'y', 'z', 'x', 'y']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nDataFrame info:")
    print(df.info())
    
    # Clean the data
    cleaned = clean_dataset(df, strategy='mean', outlier_threshold=2)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    # Validate the cleaned data
    is_valid, message = validate_data(cleaned, required_columns=['A', 'B', 'C'], min_rows=2)
    print(f"\nValidation result: {is_valid}")
    print(f"Validation message: {message}")