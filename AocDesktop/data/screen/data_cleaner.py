
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
                
                mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
                df_clean = df_clean[mask]
                
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
                    min_val = df_normalized[col].min()
                    max_val = df_normalized[col].max()
                    df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
                elif method == 'robust':
                    median = df_normalized[col].median()
                    iqr = df_normalized[col].quantile(0.75) - df_normalized[col].quantile(0.25)
                    df_normalized[col] = (df_normalized[col] - median) / iqr
                    
        self.df = df_normalized
        return self.df
    
    def handle_missing_values(self, strategy='mean', fill_value=None):
        df_filled = self.df.copy()
        
        numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
        categorical_cols = df_filled.select_dtypes(exclude=[np.number]).columns
        
        for col in numeric_cols:
            if df_filled[col].isnull().any():
                if strategy == 'mean':
                    fill_val = df_filled[col].mean()
                elif strategy == 'median':
                    fill_val = df_filled[col].median()
                elif strategy == 'mode':
                    fill_val = df_filled[col].mode()[0]
                elif strategy == 'constant' and fill_value is not None:
                    fill_val = fill_value
                else:
                    continue
                    
                df_filled[col] = df_filled[col].fillna(fill_val)
        
        for col in categorical_cols:
            if df_filled[col].isnull().any():
                if strategy == 'mode':
                    fill_val = df_filled[col].mode()[0]
                elif strategy == 'constant' and fill_value is not None:
                    fill_val = fill_value
                else:
                    fill_val = 'Unknown'
                    
                df_filled[col] = df_filled[col].fillna(fill_val)
                
        self.df = df_filled
        return self.df
    
    def get_cleaned_data(self):
        return self.df
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'current_rows': self.df.shape[0],
            'columns': self.df.shape[1],
            'missing_values': self.df.isnull().sum().sum(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(exclude=[np.number]).columns)
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'age': np.random.normal(35, 10, 100),
        'income': np.random.exponential(50000, 100),
        'score': np.random.uniform(0, 100, 100),
        'category': np.random.choice(['A', 'B', 'C', None], 100)
    }
    
    data['age'][np.random.choice(100, 5)] = np.nan
    data['income'][np.random.choice(100, 3)] = np.nan
    
    outliers = np.random.choice(100, 10)
    data['age'][outliers] = np.random.uniform(100, 150, 10)
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = create_sample_data()
    cleaner = DataCleaner(df)
    
    print("Initial summary:")
    print(cleaner.get_summary())
    
    removed = cleaner.remove_outliers_iqr(['age', 'income'])
    print(f"\nRemoved {removed} outliers")
    
    cleaner.handle_missing_values(strategy='mean')
    cleaner.normalize_data(method='zscore')
    
    print("\nFinal summary:")
    print(cleaner.get_summary())
    
    cleaned_df = cleaner.get_cleaned_data()
    print(f"\nCleaned data shape: {cleaned_df.shape}")
import pandas as pd
import numpy as np

def clean_dataset(df, key_columns=None, date_columns=None):
    """
    Clean dataset by removing duplicates, handling missing values,
    and standardizing column formats.
    """
    cleaned_df = df.copy()
    
    if key_columns:
        cleaned_df = cleaned_df.drop_duplicates(subset=key_columns)
    else:
        cleaned_df = cleaned_df.drop_duplicates()
    
    for col in cleaned_df.columns:
        if cleaned_df[col].dtype == 'object':
            cleaned_df[col] = cleaned_df[col].str.strip().str.lower()
    
    if date_columns:
        for date_col in date_columns:
            if date_col in cleaned_df.columns:
                cleaned_df[date_col] = pd.to_datetime(
                    cleaned_df[date_col], errors='coerce'
                )
    
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(
        cleaned_df[numeric_cols].median()
    )
    
    categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
    cleaned_df[categorical_cols] = cleaned_df[categorical_cols].fillna('unknown')
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    """
    if len(df) < min_rows:
        raise ValueError(f"Dataset must have at least {min_rows} rows")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'id': [1, 2, 2, 3, 4],
        'name': [' John ', 'Alice', 'alice', 'Bob', None],
        'value': [10, 20, None, 30, 40],
        'date': ['2023-01-01', '2023-01-02', 'invalid', '2023-01-03', '2023-01-04']
    })
    
    cleaned = clean_dataset(
        sample_data,
        key_columns=['id'],
        date_columns=['date']
    )
    
    print("Original dataset:")
    print(sample_data)
    print("\nCleaned dataset:")
    print(cleaned)
    
    try:
        validate_data(cleaned, required_columns=['id', 'name'], min_rows=3)
        print("\nData validation passed")
    except ValueError as e:
        print(f"\nData validation failed: {e}")