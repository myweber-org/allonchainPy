
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str or dict): Method to fill missing values. 
                                Can be 'mean', 'median', 'mode', or a dictionary of column:value pairs.
                                If None, missing values are not filled. Default is None.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        elif fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the structure and content of a DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
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

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 30, 40, 50],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_data(cleaned, required_columns=['A', 'B'], min_rows=1)
    print(f"\nValidation: {is_valid} - {message}")
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path, numeric_columns):
    df = pd.read_csv(file_path)
    
    for col in numeric_columns:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)
            df = normalize_minmax(df, col)
    
    df.to_csv('cleaned_data.csv', index=False)
    return df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 200),
        'feature_b': np.random.exponential(50, 200)
    })
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_dataset('sample_data.csv', ['feature_a', 'feature_b'])
    print(f"Original shape: {pd.read_csv('sample_data.csv').shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print("Data cleaning completed.")
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
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[col]):
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            except Exception as e:
                print(f"Warning: Could not process column '{col}': {e}")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'null_counts': {},
        'data_types': {}
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_results['is_valid'] = False
            validation_results['missing_columns'] = missing
    
    for col in df.columns:
        validation_results['null_counts'][col] = df[col].isnull().sum()
        validation_results['data_types'][col] = str(df[col].dtype)
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'id': range(1, 101),
        'value': np.random.normal(100, 15, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[10, 'value'] = 500
    df.loc[20, 'value'] = -200
    
    print("Original DataFrame shape:", df.shape)
    print("\nSample data:")
    print(df.head())
    
    cleaned_df = clean_numeric_data(df, columns=['value'])
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    
    validation = validate_dataframe(cleaned_df, required_columns=['id', 'value', 'category'])
    print("\nValidation results:")
    print(validation)
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
        
        removed_count = len(self.df) - len(df_clean)
        self.df = df_clean
        return removed_count
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_clean = self.df.copy()
        for col in columns:
            if col in df_clean.columns:
                z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                df_clean = df_clean[(z_scores < threshold) | df_clean[col].isna()]
        
        removed_count = len(self.df) - len(df_clean)
        self.df = df_clean
        return removed_count
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_normalized = self.df.copy()
        for col in columns:
            if col in df_normalized.columns:
                min_val = df_normalized[col].min()
                max_val = df_normalized[col].max()
                if max_val != min_val:
                    df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
        
        self.df = df_normalized
        return self
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_normalized = self.df.copy()
        for col in columns:
            if col in df_normalized.columns:
                mean_val = df_normalized[col].mean()
                std_val = df_normalized[col].std()
                if std_val > 0:
                    df_normalized[col] = (df_normalized[col] - mean_val) / std_val
        
        self.df = df_normalized
        return self
    
    def fill_missing_mean(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].mean())
        
        return self
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].median())
        
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_removal_stats(self):
        final_shape = self.df.shape
        rows_removed = self.original_shape[0] - final_shape[0]
        cols_removed = self.original_shape[1] - final_shape[1]
        return {
            'original_rows': self.original_shape[0],
            'current_rows': final_shape[0],
            'rows_removed': rows_removed,
            'original_cols': self.original_shape[1],
            'current_cols': final_shape[1],
            'cols_removed': cols_removed
        }