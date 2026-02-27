import pandas as pd

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

def clean_numeric_column(df, column_name):
    """
    Clean a numeric column by removing non-numeric characters.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Name of column to clean.
    
    Returns:
        pd.DataFrame: DataFrame with cleaned column.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    return df

def validate_email_format(df, email_column):
    """
    Validate email format and flag invalid entries.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        email_column (str): Name of email column.
    
    Returns:
        pd.DataFrame: DataFrame with validation results.
    """
    import re
    
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['email_valid'] = df[email_column].apply(
        lambda x: bool(re.match(pattern, str(x))) if pd.notnull(x) else False
    )
    return df

def main():
    """
    Example usage of data cleaning functions.
    """
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'name': ['Alice', 'Bob', 'Alice', 'Charlie', 'David', 'David'],
        'email': ['alice@example.com', 'bob@test', 'alice@example.com', 'charlie@domain.org', 'david@mail.com', 'david@mail.com'],
        'age': ['25', '30', '25', '35', '40', '40']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    df_clean = remove_duplicates(df, subset=['id', 'name'])
    print("After removing duplicates:")
    print(df_clean)
    print()
    
    df_clean = clean_numeric_column(df_clean, 'age')
    print("After cleaning numeric column:")
    print(df_clean)
    print()
    
    df_clean = validate_email_format(df_clean, 'email')
    print("After email validation:")
    print(df_clean)

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
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
        
        removed_count = self.original_shape[0] - clean_df.shape[0]
        self.df = clean_df
        return removed_count
    
    def normalize_data(self, columns=None, method='minmax'):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        normalized_df = self.df.copy()
        for col in columns:
            if col in normalized_df.columns:
                if method == 'minmax':
                    min_val = normalized_df[col].min()
                    max_val = normalized_df[col].max()
                    if max_val != min_val:
                        normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
                elif method == 'zscore':
                    mean_val = normalized_df[col].mean()
                    std_val = normalized_df[col].std()
                    if std_val > 0:
                        normalized_df[col] = (normalized_df[col] - mean_val) / std_val
        
        self.df = normalized_df
        return self.df
    
    def handle_missing_values(self, strategy='mean', fill_value=None):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if strategy == 'mean':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        elif strategy == 'median':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        elif strategy == 'mode':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mode().iloc[0])
        elif strategy == 'custom' and fill_value is not None:
            self.df[numeric_cols] = self.df[numeric_cols].fillna(fill_value)
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        self.df[categorical_cols] = self.df[categorical_cols].fillna('Unknown')
        
        return self.df
    
    def get_cleaned_data(self):
        return self.df
    
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

def load_and_clean_data(filepath, cleaning_steps=None):
    try:
        df = pd.read_csv(filepath)
        cleaner = DataCleaner(df)
        
        if cleaning_steps:
            for step in cleaning_steps:
                if step['action'] == 'remove_outliers':
                    cleaner.remove_outliers_iqr(**step.get('params', {}))
                elif step['action'] == 'normalize':
                    cleaner.normalize_data(**step.get('params', {}))
                elif step['action'] == 'handle_missing':
                    cleaner.handle_missing_values(**step.get('params', {}))
        
        return cleaner.get_cleaned_data(), cleaner.get_summary()
    except Exception as e:
        print(f"Error loading or cleaning data: {str(e)}")
        return None, None