import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list, optional): Column labels to consider for duplicates.
        keep (str, optional): Which duplicates to keep ('first', 'last', False).
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    if subset is None:
        subset = df.columns.tolist()
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    print(f"Removed {len(df) - len(cleaned_df)} duplicate rows")
    print(f"Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
    
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(df):
    """
    Perform basic validation checks on DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
    
    Returns:
        dict: Dictionary containing validation results.
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'null_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    
    return validation_results

def clean_numeric_columns(df, columns=None):
    """
    Clean numeric columns by converting to appropriate types.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list, optional): Specific columns to clean.
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in df.columns:
            try:
                cleaned_df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                print(f"Could not convert column {col}: {e}")
    
    return cleaned_df
import pandas as pd
import numpy as np

def clean_csv_data(file_path, fill_method='mean', output_path=None):
    """
    Load a CSV file, clean missing values, and optionally save cleaned data.
    
    Parameters:
    file_path (str): Path to input CSV file
    fill_method (str): Method for filling missing values ('mean', 'median', 'mode', 'zero')
    output_path (str): Optional path to save cleaned CSV
    
    Returns:
    pandas.DataFrame: Cleaned dataframe
    """
    
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data with shape: {df.shape}")
        
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if fill_method == 'mean':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif fill_method == 'median':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            elif fill_method == 'mode':
                for col in numeric_cols:
                    df[col] = df[col].fillna(df[col].mode()[0])
            elif fill_method == 'zero':
                df[numeric_cols] = df[numeric_cols].fillna(0)
            else:
                raise ValueError(f"Unknown fill method: {fill_method}")
            
            print(f"Missing values filled using '{fill_method}' method")
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a dataframe column using IQR method.
    
    Parameters:
    df (pandas.DataFrame): Input dataframe
    column (str): Column name to process
    multiplier (float): IQR multiplier for outlier detection
    
    Returns:
    pandas.DataFrame: Dataframe with outliers removed
    """
    if column not in df.columns:
        print(f"Column '{column}' not found in dataframe")
        return df
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    original_count = len(df)
    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    removed_count = original_count - len(df_clean)
    
    print(f"Removed {removed_count} outliers from column '{column}'")
    
    return df_clean

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, np.nan, 30, 40, 50, 60],
        'C': ['x', 'y', 'z', 'x', 'y', 'z']
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', fill_method='median', output_path='cleaned_data.csv')
    
    if cleaned_df is not None:
        print("Data cleaning completed successfully")
        print(cleaned_df.head())
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, column, multiplier=1.5):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
        return self
        
    def remove_outliers_zscore(self, column, threshold=3):
        z_scores = np.abs(stats.zscore(self.df[column]))
        self.df = self.df[z_scores < threshold]
        return self
        
    def normalize_column(self, column, method='minmax'):
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
        elif method == 'standard':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            self.df[column] = (self.df[column] - mean_val) / std_val
        return self
        
    def fill_missing(self, column, strategy='mean'):
        if strategy == 'mean':
            fill_value = self.df[column].mean()
        elif strategy == 'median':
            fill_value = self.df[column].median()
        elif strategy == 'mode':
            fill_value = self.df[column].mode()[0]
        else:
            fill_value = strategy
            
        self.df[column].fillna(fill_value, inplace=True)
        return self
        
    def get_cleaned_data(self):
        return self.df
        
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]
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

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
    return cleaned_df

if __name__ == "__main__":
    sample_data = {'values': [10, 12, 12, 13, 12, 11, 10, 100, 12, 14, 13, 10]}
    df = pd.DataFrame(sample_data)
    print("Original data:")
    print(df)
    
    result = clean_dataset(df, ['values'])
    print("\nCleaned data:")
    print(result)
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
    
    def normalize_column(self, column, method='zscore'):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
            
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise ValueError(f"Column '{column}' is not numeric")
            
        if method == 'zscore':
            self.df[f'{column}_normalized'] = stats.zscore(self.df[column])
        elif method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            if max_val != min_val:
                self.df[f'{column}_normalized'] = (self.df[column] - min_val) / (max_val - min_val)
            else:
                self.df[f'{column}_normalized'] = 0.5
        else:
            raise ValueError("Method must be 'zscore' or 'minmax'")
            
        return self.df[f'{column}_normalized']
    
    def fill_missing_with_strategy(self, columns=None, strategy='median'):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns and self.df[col].isnull().any():
                if strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 0
                else:
                    fill_value = 0
                    
                self.df[col] = self.df[col].fillna(fill_value)
                
        return self.df.isnull().sum().sum()
    
    def get_cleaned_data(self):
        return self.df.copy()
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'current_rows': self.df.shape[0],
            'columns': self.df.shape[1],
            'missing_values': self.df.isnull().sum().sum(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns)
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100),
        'score': np.random.uniform(0, 1, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(data)
    df.loc[np.random.choice(100, 5), 'value'] = np.nan
    df.loc[10:15, 'score'] = df.loc[10:15, 'score'] * 10
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Initial summary:")
    print(cleaner.get_summary())
    
    outliers_removed = cleaner.remove_outliers_iqr(['value', 'score'])
    print(f"\nRemoved {outliers_removed} outliers")
    
    missing_filled = cleaner.fill_missing_with_strategy(strategy='median')
    print(f"Filled {missing_filled} missing values")
    
    cleaner.normalize_column('value', method='minmax')
    cleaner.normalize_column('score', method='zscore')
    
    print("\nFinal summary:")
    print(cleaner.get_summary())
    
    cleaned_df = cleaner.get_cleaned_data()
    print(f"\nCleaned data shape: {cleaned_df.shape}")
    print(cleaned_df.head())