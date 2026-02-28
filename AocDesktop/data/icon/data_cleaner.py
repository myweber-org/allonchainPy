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

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns
    """
    if columns is None:
        columns = data.columns
    
    data_filled = data.copy()
    
    for column in columns:
        if column not in data.columns:
            continue
            
        if data[column].isnull().any():
            if strategy == 'mean':
                fill_value = data[column].mean()
            elif strategy == 'median':
                fill_value = data[column].median()
            elif strategy == 'mode':
                fill_value = data[column].mode()[0]
            elif strategy == 'constant':
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            data_filled[column] = data[column].fillna(fill_value)
    
    return data_filled

def validate_data(data, column, min_val=None, max_val=None, allowed_values=None):
    """
    Validate data in a column based on constraints
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    validation_results = {
        'total_rows': len(data),
        'valid_rows': 0,
        'invalid_rows': 0,
        'invalid_indices': []
    }
    
    for idx, value in enumerate(data[column]):
        valid = True
        
        if min_val is not None and value < min_val:
            valid = False
        if max_val is not None and value > max_val:
            valid = False
        if allowed_values is not None and value not in allowed_values:
            valid = False
        
        if valid:
            validation_results['valid_rows'] += 1
        else:
            validation_results['invalid_rows'] += 1
            validation_results['invalid_indices'].append(idx)
    
    return validation_results