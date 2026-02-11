
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_clean = self.df.copy()
        for col in columns:
            if col in self.df.columns:
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                mask = z_scores < threshold
                df_clean = df_clean[mask | self.df[col].isna()]
        
        self.df = df_clean.reset_index(drop=True)
        return self
        
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns:
                col_min = self.df[col].min()
                col_max = self.df[col].max()
                if col_max > col_min:
                    self.df[col] = (self.df[col] - col_min) / (col_max - col_min)
        
        return self
        
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns:
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)
        
        return self
        
    def get_cleaned_data(self):
        return self.df
        
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]

def clean_dataset(data_path):
    df = pd.read_csv(data_path)
    cleaner = DataCleaner(df)
    
    cleaner.remove_outliers_zscore()
    cleaner.fill_missing_median()
    cleaner.normalize_minmax()
    
    print(f"Removed {cleaner.get_removed_count()} outliers")
    print(f"Final shape: {cleaner.df.shape}")
    
    return cleaner.get_cleaned_data()
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        column_mapping (dict, optional): Dictionary mapping old column names to new ones.
        drop_duplicates (bool): Whether to remove duplicate rows.
        normalize_text (bool): Whether to normalize text columns (strip, lower, remove extra spaces).
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    
    if normalize_text:
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[col] = cleaned_df[col].astype(str).apply(lambda x: re.sub(r'\s+', ' ', x.strip().lower()))
    
    return cleaned_df

def validate_email(email_series):
    """
    Validate email addresses in a pandas Series.
    
    Args:
        email_series (pd.Series): Series containing email addresses.
    
    Returns:
        pd.Series: Boolean Series indicating valid emails.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return email_series.str.match(pattern)

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to process.
        multiplier (float): IQR multiplier for outlier detection.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]