
import pandas as pd
import numpy as np
from typing import List, Optional

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape

    def remove_duplicates(self, subset: Optional[List[str]] = None) -> pd.DataFrame:
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep='first')
        removed = initial_count - len(self.df)
        print(f"Removed {removed} duplicate rows")
        return self.df

    def normalize_column(self, column_name: str, method: str = 'minmax') -> pd.DataFrame:
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame")

        if method == 'minmax':
            col_min = self.df[column_name].min()
            col_max = self.df[column_name].max()
            if col_max != col_min:
                self.df[f"{column_name}_normalized"] = (self.df[column_name] - col_min) / (col_max - col_min)
            else:
                self.df[f"{column_name}_normalized"] = 0.5
        elif method == 'zscore':
            col_mean = self.df[column_name].mean()
            col_std = self.df[column_name].std()
            if col_std > 0:
                self.df[f"{column_name}_normalized"] = (self.df[column_name] - col_mean) / col_std
            else:
                self.df[f"{column_name}_normalized"] = 0
        else:
            raise ValueError("Method must be 'minmax' or 'zscore'")
        
        return self.df

    def handle_missing_values(self, strategy: str = 'mean', columns: Optional[List[str]] = None) -> pd.DataFrame:
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col not in self.df.columns:
                continue
                
            if strategy == 'mean':
                fill_value = self.df[col].mean()
            elif strategy == 'median':
                fill_value = self.df[col].median()
            elif strategy == 'mode':
                fill_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 0
            elif strategy == 'drop':
                self.df = self.df.dropna(subset=[col])
                continue
            else:
                raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")
            
            missing_count = self.df[col].isna().sum()
            if missing_count > 0:
                self.df[col] = self.df[col].fillna(fill_value)
                print(f"Filled {missing_count} missing values in column '{col}' with {strategy}: {fill_value}")
        
        return self.df

    def get_summary(self) -> dict:
        return {
            'original_shape': self.original_shape,
            'current_shape': self.df.shape,
            'columns': list(self.df.columns),
            'missing_values': self.df.isna().sum().to_dict(),
            'data_types': self.df.dtypes.astype(str).to_dict()
        }

def clean_dataset(file_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    cleaner = DataCleaner(df)
    
    cleaner.remove_duplicates()
    cleaner.handle_missing_values(strategy='median')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        cleaner.normalize_column(col, method='minmax')
    
    summary = cleaner.get_summary()
    print(f"Data cleaning complete. Original shape: {summary['original_shape']}, Final shape: {summary['current_shape']}")
    
    if output_path:
        cleaner.df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
    
    return cleaner.df
import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    subset (list, optional): Columns to consider for duplicates
    keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    if subset is None:
        subset = df.columns.tolist()
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_numeric_columns(df, columns=None):
    """
    Clean numeric columns by converting to appropriate types and handling NaN.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list, optional): Specific columns to clean
    
    Returns:
    pd.DataFrame: DataFrame with cleaned numeric columns
    """
    if df.empty:
        return df
    
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
    else:
        numeric_cols = [col for col in columns if col in df.columns]
    
    cleaned_df = df.copy()
    
    for col in numeric_cols:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    required_columns (list, optional): Columns that must be present
    
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

def clean_data_pipeline(df, cleaning_steps=None):
    """
    Execute a series of data cleaning steps.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    cleaning_steps (list, optional): List of cleaning functions to apply
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if cleaning_steps is None:
        cleaning_steps = [
            lambda x: remove_duplicates(x),
            lambda x: clean_numeric_columns(x)
        ]
    
    cleaned_df = df.copy()
    
    for step in cleaning_steps:
        cleaned_df = step(cleaned_df)
    
    return cleaned_df
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
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
    Clean numeric columns by removing outliers from specified columns or all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[col]):
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            except Exception as e:
                print(f"Warning: Could not clean column '{col}': {e}")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'id': range(1, 21),
        'value': [10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                  21, 22, 23, 24, 25, 100, 120, 130, 140, 150]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original data:")
    print(df.describe())
    
    cleaned_df = clean_numeric_data(df, columns=['value'])
    print("\nCleaned data:")
    print(cleaned_df.describe())
    print(f"\nRemoved {len(df) - len(cleaned_df)} outliers")