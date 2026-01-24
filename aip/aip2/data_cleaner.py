import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range (IQR) method.
    
    Args:
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
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to analyze
    
    Returns:
        dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def process_numerical_data(df, columns):
    """
    Process multiple numerical columns by removing outliers.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to process
    
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    processed_df = df.copy()
    
    for column in columns:
        if column in processed_df.columns and pd.api.types.is_numeric_dtype(processed_df[column]):
            processed_df = remove_outliers_iqr(processed_df, column)
    
    return processed_df

if __name__ == "__main__":
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 27, 28, 29, 30, 100],
        'humidity': [45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
        'pressure': [1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nSummary statistics for temperature:")
    print(calculate_summary_statistics(df, 'temperature'))
    
    cleaned_df = remove_outliers_iqr(df, 'temperature')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nSummary statistics after cleaning:")
    print(calculate_summary_statistics(cleaned_df, 'temperature'))
import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.df.columns
            
        for col in columns:
            if self.df[col].dtype in ['int64', 'float64']:
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0]
                else:
                    fill_value = strategy
                    
                self.df[col].fillna(fill_value, inplace=True)
            else:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                
        return self.df
    
    def remove_outliers_iqr(self, columns=None, threshold=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_clean = self.df.copy()
        
        for col in columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
            df_clean = df_clean[mask]
            
        self.df = df_clean
        return self.df
    
    def get_cleaning_report(self):
        removed_rows = self.original_shape[0] - self.df.shape[0]
        removed_cols = self.original_shape[1] - self.df.shape[1]
        
        report = {
            'original_shape': self.original_shape,
            'cleaned_shape': self.df.shape,
            'rows_removed': removed_rows,
            'columns_removed': removed_cols,
            'missing_values_remaining': self.df.isnull().sum().sum()
        }
        
        return report
    
    def save_cleaned_data(self, filepath):
        self.df.to_csv(filepath, index=False)
        return f"Data saved to {filepath}"import pandas as pd

def clean_dataframe(df, column_name):
    """
    Clean a specific column in a pandas DataFrame.
    Removes duplicates, strips whitespace, and converts to lowercase.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")
    
    # Remove duplicates
    df = df.drop_duplicates(subset=[column_name])
    
    # Strip whitespace and convert to lowercase
    df[column_name] = df[column_name].astype(str).str.strip().str.lower()
    
    # Drop rows where the column is empty after cleaning
    df = df[df[column_name] != '']
    
    return df.reset_index(drop=True)