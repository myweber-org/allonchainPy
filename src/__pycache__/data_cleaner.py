import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing='mean'):
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
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            elif fill_missing == 'median':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            mode_val = cleaned_df[col].mode()
            if not mode_val.empty:
                cleaned_df[col] = cleaned_df[col].fillna(mode_val[0])
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate a DataFrame for required columns and minimum row count.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names. Default is None.
    min_rows (int): Minimum number of rows required. Default is 1.
    
    Returns:
    tuple: (bool, str) indicating validation success and message.
    """
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame validation passed"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', 'z', 'z']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataframe(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B'])
    print(f"\nValidation: {is_valid} - {message}")
import pandas as pd
import numpy as np

def clean_missing_data(file_path, strategy='mean', columns=None):
    """
    Clean missing data from a CSV file using specified strategy.
    
    Args:
        file_path (str): Path to the CSV file
        strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
        columns (list): List of columns to apply cleaning to, None for all columns
    
    Returns:
        pandas.DataFrame: Cleaned dataframe
    """
    
    try:
        df = pd.read_csv(file_path)
        
        if columns is None:
            columns = df.columns
        
        for column in columns:
            if column in df.columns:
                if strategy == 'mean':
                    df[column].fillna(df[column].mean(), inplace=True)
                elif strategy == 'median':
                    df[column].fillna(df[column].median(), inplace=True)
                elif strategy == 'mode':
                    df[column].fillna(df[column].mode()[0], inplace=True)
                elif strategy == 'drop':
                    df.dropna(subset=[column], inplace=True)
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def detect_outliers(df, column, method='iqr', threshold=1.5):
    """
    Detect outliers in a dataframe column.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        column (str): Column name to check for outliers
        method (str): Method for outlier detection ('iqr' or 'zscore')
        threshold (float): Threshold for outlier detection
    
    Returns:
        list: Indices of outliers
    """
    
    if column not in df.columns:
        return []
    
    data = df[column].dropna()
    
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = data[(data < lower_bound) | (data > upper_bound)]
    
    elif method == 'zscore':
        mean = data.mean()
        std = data.std()
        z_scores = (data - mean) / std
        outliers = data[abs(z_scores) > threshold]
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return outliers.index.tolist()

def save_cleaned_data(df, output_path):
    """
    Save cleaned dataframe to CSV file.
    
    Args:
        df (pandas.DataFrame): Cleaned dataframe
        output_path (str): Path to save the cleaned data
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    try:
        df.to_csv(output_path, index=False)
        return True
    except Exception as e:
        print(f"Error saving file: {str(e)}")
        return False

if __name__ == "__main__":
    # Example usage
    input_file = "data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_missing_data(input_file, strategy='mean')
    
    if cleaned_df is not None:
        print(f"Data cleaned successfully. Shape: {cleaned_df.shape}")
        
        # Detect outliers in numeric columns
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            outliers = detect_outliers(cleaned_df, col)
            if outliers:
                print(f"Found {len(outliers)} outliers in column '{col}'")
        
        # Save cleaned data
        if save_cleaned_data(cleaned_df, output_file):
            print(f"Cleaned data saved to {output_file}")
import pandas as pd
import numpy as np
from typing import Optional

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[list] = None) -> pd.DataFrame:
        self.df = self.df.drop_duplicates(subset=subset, keep='first')
        return self.df
    
    def handle_missing_values(self, strategy: str = 'mean', fill_value: Optional[float] = None) -> pd.DataFrame:
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if strategy == 'mean':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        elif strategy == 'median':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        elif strategy == 'mode':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mode().iloc[0])
        elif strategy == 'constant' and fill_value is not None:
            self.df[numeric_cols] = self.df[numeric_cols].fillna(fill_value)
        elif strategy == 'drop':
            self.df = self.df.dropna(subset=numeric_cols)
        
        return self.df
    
    def normalize_numeric(self, method: str = 'minmax') -> pd.DataFrame:
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if method == 'minmax':
            for col in numeric_cols:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            for col in numeric_cols:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
        
        return self.df
    
    def get_cleaning_report(self) -> dict:
        current_shape = self.df.shape
        return {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'current_rows': current_shape[0],
            'current_columns': current_shape[1],
            'rows_removed': self.original_shape[0] - current_shape[0],
            'missing_values': int(self.df.isnull().sum().sum()),
            'duplicates_removed': self.original_shape[0] - current_shape[0]
        }
    
    def save_cleaned_data(self, filepath: str) -> None:
        self.df.to_csv(filepath, index=False)

def load_and_clean_csv(filepath: str, cleaning_steps: list = None) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    cleaner = DataCleaner(df)
    
    if cleaning_steps:
        for step in cleaning_steps:
            if step['action'] == 'remove_duplicates':
                cleaner.remove_duplicates(step.get('subset'))
            elif step['action'] == 'handle_missing':
                cleaner.handle_missing_values(
                    strategy=step.get('strategy', 'mean'),
                    fill_value=step.get('fill_value')
                )
            elif step['action'] == 'normalize':
                cleaner.normalize_numeric(method=step.get('method', 'minmax'))
    
    return cleaner.df