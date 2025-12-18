import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, normalize_columns=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing column names.
    
    Args:
        df: pandas DataFrame to clean
        drop_duplicates: If True, remove duplicate rows
        normalize_columns: If True, normalize column names to lowercase with underscores
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if normalize_columns:
        cleaned_df.columns = [
            col.strip().lower().replace(' ', '_').replace('-', '_')
            for col in cleaned_df.columns
        ]
        print("Column names normalized")
    
    return cleaned_df

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in specified columns using different strategies.
    
    Args:
        df: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'drop'
        columns: List of columns to process, or None for all numeric columns
    
    Returns:
        DataFrame with missing values handled
    """
    df_copy = df.copy()
    
    if columns is None:
        columns = df_copy.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df_copy.columns:
            if strategy == 'mean':
                df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
            elif strategy == 'median':
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())
            elif strategy == 'mode':
                df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])
            elif strategy == 'drop':
                df_copy = df_copy.dropna(subset=[col])
    
    return df_copy

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    sample_data = {
        'Name': ['Alice', 'Bob', 'Alice', 'Charlie', None],
        'Age': [25, 30, 25, 35, 40],
        'Score': [85.5, 92.0, 85.5, 78.5, 90.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df)
    print("Cleaned DataFrame:")
    print(cleaned)
    
    validated, message = validate_dataframe(cleaned, required_columns=['name', 'age'])
    print(f"\nValidation: {message}")
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
    
    def normalize_column(self, column, method='minmax'):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
            
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise ValueError(f"Column '{column}' is not numeric")
            
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            if max_val == min_val:
                self.df[f'{column}_normalized'] = 0.5
            else:
                self.df[f'{column}_normalized'] = (self.df[column] - min_val) / (max_val - min_val)
                
        elif method == 'zscore':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            if std_val == 0:
                self.df[f'{column}_normalized'] = 0
            else:
                self.df[f'{column}_normalized'] = (self.df[column] - mean_val) / std_val
                
        elif method == 'robust':
            median_val = self.df[column].median()
            iqr_val = stats.iqr(self.df[column])
            if iqr_val == 0:
                self.df[f'{column}_normalized'] = 0
            else:
                self.df[f'{column}_normalized'] = (self.df[column] - median_val) / iqr_val
                
        else:
            raise ValueError(f"Unknown normalization method: {method}")
            
        return self.df[f'{column}_normalized']
    
    def fill_missing_values(self, strategy='mean', custom_value=None):
        df_filled = self.df.copy()
        
        for col in self.df.columns:
            if self.df[col].isnull().any():
                if strategy == 'mean' and pd.api.types.is_numeric_dtype(self.df[col]):
                    fill_value = self.df[col].mean()
                elif strategy == 'median' and pd.api.types.is_numeric_dtype(self.df[col]):
                    fill_value = self.df[col].median()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0] if not self.df[col].mode().empty else None
                elif strategy == 'custom' and custom_value is not None:
                    fill_value = custom_value
                elif strategy == 'forward':
                    df_filled[col] = self.df[col].ffill()
                    continue
                elif strategy == 'backward':
                    df_filled[col] = self.df[col].bfill()
                    continue
                else:
                    continue
                    
                if fill_value is not None:
                    df_filled[col] = self.df[col].fillna(fill_value)
                    
        self.df = df_filled
        return self.df.isnull().sum().sum()
    
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

def create_sample_data():
    np.random.seed(42)
    data = {
        'id': range(100),
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(data)
    
    df.loc[np.random.choice(100, 5), 'feature_a'] = np.nan
    df.loc[10:15, 'feature_b'] = df['feature_b'].max() * 10
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Initial data shape:", cleaner.original_shape)
    print("Missing values before:", sample_df.isnull().sum().sum())
    
    removed = cleaner.remove_outliers_iqr(['feature_a', 'feature_b'])
    print(f"Removed {removed} outliers")
    
    missing_filled = cleaner.fill_missing_values(strategy='mean')
    print(f"Filled {missing_filled} missing values")
    
    cleaner.normalize_column('feature_a', method='minmax')
    cleaner.normalize_column('feature_b', method='zscore')
    
    summary = cleaner.get_summary()
    print("\nCleaning Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    cleaned_df = cleaner.get_cleaned_data()
    print(f"\nFinal data shape: {cleaned_df.shape}")