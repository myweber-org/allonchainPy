
import pandas as pd
import numpy as np
from scipy import stats

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
    
    df = df.dropna()
    return df

if __name__ == "__main__":
    cleaned_data = clean_dataset('raw_data.csv', ['age', 'income', 'score'])
    cleaned_data.to_csv('cleaned_data.csv', index=False)
    print(f"Data cleaning complete. Remaining records: {len(cleaned_data)}")import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to remove duplicate rows.
        fill_missing (str): Strategy for filling missing values.
            Options: 'mean', 'median', 'mode', or 'drop'.
    
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
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Args:
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
    
    is_valid, message = validate_dataframe(cleaned)
    print(f"\nValidation: {message}")import pandas as pd
import numpy as np

def clean_dataset(df, numeric_columns=None, strategy='median', outlier_threshold=3):
    """
    Clean dataset by handling missing values and removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    numeric_columns (list): List of numeric column names
    strategy (str): Imputation strategy ('mean', 'median', 'mode')
    outlier_threshold (float): Z-score threshold for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = df.copy()
    
    # Handle missing values
    for col in numeric_columns:
        if df_clean[col].isnull().any():
            if strategy == 'mean':
                fill_value = df_clean[col].mean()
            elif strategy == 'median':
                fill_value = df_clean[col].median()
            elif strategy == 'mode':
                fill_value = df_clean[col].mode()[0]
            else:
                fill_value = 0
            
            df_clean[col].fillna(fill_value, inplace=True)
    
    # Remove outliers using Z-score method
    z_scores = np.abs((df_clean[numeric_columns] - df_clean[numeric_columns].mean()) / 
                      df_clean[numeric_columns].std())
    
    outlier_mask = (z_scores < outlier_threshold).all(axis=1)
    df_clean = df_clean[outlier_mask].reset_index(drop=True)
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Parameters:
    df (pd.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

def calculate_statistics(df, numeric_columns=None):
    """
    Calculate basic statistics for numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    numeric_columns (list): List of numeric column names
    
    Returns:
    pd.DataFrame: Statistics dataframe
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    stats = pd.DataFrame({
        'mean': df[numeric_columns].mean(),
        'median': df[numeric_columns].median(),
        'std': df[numeric_columns].std(),
        'min': df[numeric_columns].min(),
        'max': df[numeric_columns].max(),
        'missing': df[numeric_columns].isnull().sum()
    })
    
    return stats

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    data = {
        'feature_a': np.random.randn(100),
        'feature_b': np.random.randn(100) * 2 + 5,
        'feature_c': np.random.randn(100) * 0.5 + 10
    }
    
    # Introduce some missing values and outliers
    sample_df = pd.DataFrame(data)
    sample_df.loc[10:15, 'feature_a'] = np.nan
    sample_df.loc[95, 'feature_b'] = 50  # Outlier
    
    print("Original DataFrame shape:", sample_df.shape)
    print("\nMissing values:")
    print(sample_df.isnull().sum())
    
    # Clean the data
    cleaned_df = clean_dataset(sample_df, strategy='median')
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("\nCleaned statistics:")
    stats = calculate_statistics(cleaned_df)
    print(stats)
import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_columns = df.columns.tolist()
        
    def remove_outliers_iqr(self, column, multiplier=1.5):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
            
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        mask = (self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)
        self.df = self.df[mask].reset_index(drop=True)
        return self
        
    def normalize_column(self, column, method='minmax'):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
            
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            if max_val != min_val:
                self.df[f'{column}_normalized'] = (self.df[column] - min_val) / (max_val - min_val)
            else:
                self.df[f'{column}_normalized'] = 0.5
                
        elif method == 'zscore':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            if std_val > 0:
                self.df[f'{column}_normalized'] = (self.df[column] - mean_val) / std_val
            else:
                self.df[f'{column}_normalized'] = 0
                
        return self
        
    def fill_missing(self, column, strategy='mean'):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
            
        if strategy == 'mean':
            fill_value = self.df[column].mean()
        elif strategy == 'median':
            fill_value = self.df[column].median()
        elif strategy == 'mode':
            fill_value = self.df[column].mode()[0]
        elif isinstance(strategy, (int, float)):
            fill_value = strategy
        else:
            raise ValueError("Invalid strategy. Use 'mean', 'median', 'mode', or a numeric value")
            
        self.df[column] = self.df[column].fillna(fill_value)
        return self
        
    def get_cleaned_data(self):
        return self.df
        
    def get_summary(self):
        summary = {
            'original_rows': len(self.df),
            'remaining_rows': len(self.df),
            'columns': self.df.columns.tolist(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict()
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'feature_c': np.random.randint(1, 100, 100)
    }
    
    df = pd.DataFrame(data)
    df.loc[np.random.choice(df.index, 10), 'feature_a'] = np.nan
    df.loc[np.random.choice(df.index, 5), 'feature_b'] = np.nan
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    cleaner.fill_missing('feature_a', 'mean')
    cleaner.fill_missing('feature_b', 'median')
    cleaner.remove_outliers_iqr('feature_c')
    cleaner.normalize_column('feature_a', 'minmax')
    cleaner.normalize_column('feature_b', 'zscore')
    
    cleaned_df = cleaner.get_cleaned_data()
    summary = cleaner.get_summary()
    
    print(f"Original shape: {sample_df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Missing values after cleaning: {cleaned_df.isnull().sum().sum()}")import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Method to fill missing values. Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
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

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (bool, str) indicating success and message.
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"