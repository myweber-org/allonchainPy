
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
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
            mode_value = cleaned_df[col].mode()
            if not mode_value.empty:
                cleaned_df[col].fillna(mode_value[0], inplace=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
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
import pandas as pd
import numpy as np
from pathlib import Path

class CSVDataCleaner:
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.df = None
        self.cleaning_report = {}
        
    def load_data(self):
        try:
            self.df = pd.read_csv(self.filepath)
            self.cleaning_report['original_rows'] = len(self.df)
            self.cleaning_report['original_columns'] = len(self.df.columns)
            return True
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def remove_duplicates(self):
        if self.df is not None:
            initial_count = len(self.df)
            self.df.drop_duplicates(inplace=True)
            removed = initial_count - len(self.df)
            self.cleaning_report['duplicates_removed'] = removed
            return removed
        return 0
    
    def handle_missing_values(self, strategy='mean', custom_value=None):
        if self.df is None:
            return
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        missing_counts = {}
        
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            if missing_count > 0:
                missing_counts[col] = missing_count
                
                if col in numeric_cols:
                    if strategy == 'mean':
                        fill_value = self.df[col].mean()
                    elif strategy == 'median':
                        fill_value = self.df[col].median()
                    elif strategy == 'custom' and custom_value is not None:
                        fill_value = custom_value
                    else:
                        fill_value = 0
                    self.df[col].fillna(fill_value, inplace=True)
                else:
                    self.df[col].fillna('Unknown', inplace=True)
        
        self.cleaning_report['missing_values_handled'] = missing_counts
    
    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if self.df is None or self.df.empty:
            return 0
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        initial_count = len(self.df)
        outliers_removed = 0
        
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                before = len(self.df)
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
                outliers_removed += (before - len(self.df))
        
        self.cleaning_report['outliers_removed'] = outliers_removed
        return outliers_removed
    
    def standardize_column_names(self):
        if self.df is not None:
            new_columns = {}
            for col in self.df.columns:
                new_name = col.strip().lower().replace(' ', '_').replace('-', '_')
                new_columns[col] = new_name
            self.df.rename(columns=new_columns, inplace=True)
            self.cleaning_report['columns_renamed'] = new_columns
    
    def save_cleaned_data(self, output_path=None):
        if self.df is None:
            return False
        
        if output_path is None:
            output_path = self.filepath.parent / f"cleaned_{self.filepath.name}"
        
        try:
            self.df.to_csv(output_path, index=False)
            self.cleaning_report['output_file'] = str(output_path)
            self.cleaning_report['final_rows'] = len(self.df)
            self.cleaning_report['final_columns'] = len(self.df.columns)
            return True
        except Exception as e:
            print(f"Error saving file: {e}")
            return False
    
    def get_cleaning_report(self):
        return self.cleaning_report
    
    def clean_pipeline(self, output_path=None):
        self.load_data()
        self.remove_duplicates()
        self.handle_missing_values(strategy='mean')
        self.remove_outliers_iqr()
        self.standardize_column_names()
        self.save_cleaned_data(output_path)
        return self.get_cleaning_report()

def example_usage():
    cleaner = CSVDataCleaner('raw_data.csv')
    report = cleaner.clean_pipeline('cleaned_data.csv')
    
    print("Cleaning Report:")
    for key, value in report.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    example_usage()
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
    
    return filtered_df

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
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            except Exception as e:
                print(f"Warning: Could not clean column '{col}': {e}")
    
    return cleaned_df

def calculate_statistics(df):
    """
    Calculate basic statistics for numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    pd.DataFrame: Statistics summary
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return pd.DataFrame()
    
    stats = {
        'mean': numeric_df.mean(),
        'median': numeric_df.median(),
        'std': numeric_df.std(),
        'min': numeric_df.min(),
        'max': numeric_df.max(),
        'count': numeric_df.count()
    }
    
    return pd.DataFrame(stats)