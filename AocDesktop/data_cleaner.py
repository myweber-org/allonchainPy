
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
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def z_score_normalize(data, column):
    """
    Normalize data using z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    normalized = (data[column] - mean_val) / std_val
    return normalized

def clean_dataset(df, numeric_columns=None, outlier_factor=1.5):
    """
    Clean dataset by removing outliers and normalizing numeric columns
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    removal_stats = {}
    
    for col in numeric_columns:
        if col in df.columns:
            cleaned_df, removed = remove_outliers_iqr(cleaned_df, col, outlier_factor)
            removal_stats[col] = removed
            
            cleaned_df[f'{col}_normalized'] = normalize_minmax(cleaned_df, col)
            cleaned_df[f'{col}_zscore'] = z_score_normalize(cleaned_df, col)
    
    return cleaned_df, removal_stats

def validate_data(df, required_columns, min_rows=10):
    """
    Validate dataset structure and content
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if len(df) < min_rows:
        raise ValueError(f"Dataset has only {len(df)} rows, minimum required is {min_rows}")
    
    null_counts = df.isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0].index.tolist()
    
    validation_result = {
        'row_count': len(df),
        'column_count': len(df.columns),
        'missing_columns': missing_columns,
        'columns_with_nulls': columns_with_nulls,
        'null_counts': null_counts.to_dict(),
        'is_valid': len(missing_columns) == 0 and len(df) >= min_rows
    }
    
    return validation_result
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to remove duplicate rows
        fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values")
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        print(f"Filled missing numeric values with {fill_missing}")
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown')
        print("Filled missing categorical values with mode")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append("DataFrame is empty")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_cols}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            validation_results['warnings'].append(f"Column '{col}' contains missing values")
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, np.nan, 40.1, 50.0],
        'category': ['A', 'B', 'B', 'C', None, 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    validation = validate_dataframe(cleaned, required_columns=['id', 'value'])
    print("\nValidation Results:")
    print(validation)
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, columns):
    """Remove outliers using IQR method."""
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def normalize_data(df, columns, method='minmax'):
    """Normalize specified columns using min-max or z-score normalization."""
    df_norm = df.copy()
    for col in columns:
        if method == 'minmax':
            df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
        elif method == 'zscore':
            df_norm[col] = (df_norm[col] - df_norm[col].mean()) / df_norm[col].std()
    return df_norm

def handle_missing_values(df, strategy='mean'):
    """Handle missing values using specified strategy."""
    df_filled = df.copy()
    for col in df_filled.columns:
        if df_filled[col].isnull().any():
            if strategy == 'mean':
                df_filled[col].fillna(df_filled[col].mean(), inplace=True)
            elif strategy == 'median':
                df_filled[col].fillna(df_filled[col].median(), inplace=True)
            elif strategy == 'mode':
                df_filled[col].fillna(df_filled[col].mode()[0], inplace=True)
    return df_filled

def clean_dataset(filepath, numeric_columns, normalization_method='minmax'):
    """Complete data cleaning pipeline."""
    df = load_dataset(filepath)
    df = handle_missing_values(df)
    df = remove_outliers_iqr(df, numeric_columns)
    df = normalize_data(df, numeric_columns, method=normalization_method)
    return df

if __name__ == "__main__":
    cleaned_data = clean_dataset('raw_data.csv', ['age', 'income', 'score'])
    cleaned_data.to_csv('cleaned_data.csv', index=False)
    print("Data cleaning completed. Cleaned data saved to 'cleaned_data.csv'")
import pandas as pd

def clean_dataset(df, remove_duplicates=True, fill_method='drop'):
    """
    Clean a pandas DataFrame by handling missing values and optionally removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    remove_duplicates (bool): If True, remove duplicate rows.
    fill_method (str): Method to handle missing values: 'drop' to remove rows,
                       'fill_mean' to fill numeric columns with mean,
                       'fill_median' to fill numeric columns with median,
                       'fill_mode' to fill with mode for categorical columns.
    
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
    elif fill_method == 'fill_mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                mode_val = cleaned_df[col].mode()
                if not mode_val.empty:
                    cleaned_df[col] = cleaned_df[col].fillna(mode_val.iloc[0])
    
    # Remove duplicates if requested
    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = {
#         'A': [1, 2, None, 4, 1],
#         'B': [5, None, 7, 8, 5],
#         'C': ['x', 'y', 'z', None, 'x']
#     }
#     df = pd.DataFrame(sample_data)
#     print("Original DataFrame:")
#     print(df)
#     
#     cleaned = clean_dataset(df, remove_duplicates=True, fill_method='fill_mean')
#     print("\nCleaned DataFrame:")
#     print(cleaned)
#     
#     is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B'])
#     print(f"\nValidation: {is_valid}, Message: {message}")
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
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
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
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
                    
        return self
        
    def standardize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
                    
        return self
        
    def handle_missing_mean(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                mean_val = self.df[col].mean()
                self.df[col] = self.df[col].fillna(mean_val)
                
        return self
        
    def get_cleaned_data(self):
        return self.df
        
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 200, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    data['feature_a'][np.random.choice(1000, 20)] = np.nan
    data['feature_b'][np.random.choice(1000, 30)] = np.nan
    
    outliers = np.random.choice(1000, 10, replace=False)
    data['feature_a'][outliers] = data['feature_a'][outliers] * 5
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = create_sample_data()
    print(f"Original data shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    cleaner = DataCleaner(df)
    cleaned_df = (cleaner
                 .handle_missing_mean()
                 .remove_outliers_iqr(threshold=1.5)
                 .standardize_zscore()
                 .get_cleaned_data())
    
    print(f"\nCleaned data shape: {cleaned_df.shape}")
    print(f"Rows removed: {cleaner.get_removed_count()}")
    print(f"\nCleaned statistics:")
    print(cleaned_df.describe())