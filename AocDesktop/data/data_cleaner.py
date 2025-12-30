import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to remove duplicate rows
        fill_missing (str or dict): Method to fill missing values:
            - 'mean': Fill with column mean
            - 'median': Fill with column median
            - 'mode': Fill with column mode
            - dict: Column-specific fill values
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Handle missing values
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

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
        min_rows (int): Minimum number of rows required
    
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
    
    return True, "DataFrame is valid"

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = {
#         'A': [1, 2, 2, None, 5],
#         'B': [10, None, 30, 40, 50],
#         'C': ['x', 'y', 'x', 'z', None]
#     }
#     df = pd.DataFrame(sample_data)
#     cleaned = clean_dataframe(df, fill_missing='mean')
#     print("Original DataFrame:")
#     print(df)
#     print("\nCleaned DataFrame:")
#     print(cleaned)
import pandas as pd
import numpy as np
from typing import Optional, Dict, List

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def handle_missing_values(self, strategy: str = 'mean', columns: Optional[List[str]] = None) -> 'DataCleaner':
        if columns is None:
            columns = self.df.columns
            
        for col in columns:
            if col in self.df.columns:
                if strategy == 'mean' and self.df[col].dtype in ['int64', 'float64']:
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif strategy == 'median' and self.df[col].dtype in ['int64', 'float64']:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                elif strategy == 'mode':
                    self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else 0, inplace=True)
                elif strategy == 'drop':
                    self.df.dropna(subset=[col], inplace=True)
                else:
                    self.df[col].fillna(0, inplace=True)
        return self
    
    def convert_types(self, type_map: Dict[str, str]) -> 'DataCleaner':
        for col, dtype in type_map.items():
            if col in self.df.columns:
                try:
                    if dtype == 'datetime':
                        self.df[col] = pd.to_datetime(self.df[col])
                    else:
                        self.df[col] = self.df[col].astype(dtype)
                except Exception as e:
                    print(f"Error converting column {col} to {dtype}: {e}")
        return self
    
    def remove_outliers(self, column: str, method: str = 'iqr', threshold: float = 1.5) -> 'DataCleaner':
        if column not in self.df.columns:
            return self
            
        if method == 'iqr':
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
        
        return self
    
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df
    
    def get_cleaning_report(self) -> Dict:
        return {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': self.df.shape[0],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'missing_values_remaining': self.df.isnull().sum().sum()
        }

def load_and_clean_csv(filepath: str, cleaning_steps: Optional[Dict] = None) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    cleaner = DataCleaner(df)
    
    if cleaning_steps:
        if 'missing_values' in cleaning_steps:
            cleaner.handle_missing_values(**cleaning_steps['missing_values'])
        if 'type_conversion' in cleaning_steps:
            cleaner.convert_types(cleaning_steps['type_conversion'])
        if 'outlier_removal' in cleaning_steps:
            for outlier_config in cleaning_steps['outlier_removal']:
                cleaner.remove_outliers(**outlier_config)
    
    return cleaner.get_cleaned_data()import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (str): Method to fill missing values. Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()
    
    if fill_missing == 'drop':
        df_clean = df_clean.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = df_clean.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            elif fill_missing == 'median':
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
    elif fill_missing == 'mode':
        for col in df_clean.columns:
            df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else None, inplace=True)
    
    return df_clean

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate a DataFrame for required columns and minimum row count.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
        min_rows (int): Minimum number of rows required.
    
    Returns:
        tuple: (bool, str) indicating success and error message if any.
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Validation passed"
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    cleaned_df.reset_index(drop=True, inplace=True)
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 200),
        'B': np.random.exponential(50, 200),
        'C': np.random.uniform(0, 1, 200)
    })
    sample_data.loc[10:15, 'A'] = 500
    sample_data.loc[20:25, 'B'] = 1000
    
    numeric_cols = ['A', 'B']
    result = clean_dataset(sample_data, numeric_cols)
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {result.shape}")
    print(f"Rows removed: {len(sample_data) - len(result)}")