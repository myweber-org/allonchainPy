def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np
from typing import Optional

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self) -> 'DataCleaner':
        self.df = self.df.drop_duplicates()
        return self
        
    def handle_missing_values(self, strategy: str = 'mean', columns: Optional[list] = None) -> 'DataCleaner':
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns and self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'mode':
                    fill_value = self.df[col].mode()[0]
                elif strategy == 'drop':
                    self.df = self.df.dropna(subset=[col])
                    continue
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                    
                self.df[col] = self.df[col].fillna(fill_value)
                
        return self
        
    def remove_outliers_iqr(self, columns: Optional[list] = None, multiplier: float = 1.5) -> 'DataCleaner':
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        for col in columns:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
                
        return self
        
    def normalize_column(self, column: str) -> 'DataCleaner':
        if column in self.df.columns:
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            if max_val != min_val:
                self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
        return self
        
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df.copy()
        
    def get_cleaning_report(self) -> dict:
        return {
            'original_rows': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_rows': self.df.shape[0],
            'cleaned_columns': self.df.shape[1],
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'columns_removed': self.original_shape[1] - self.df.shape[1]
        }

def load_and_clean_csv(filepath: str, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    cleaner = DataCleaner(df)
    
    if 'remove_duplicates' in kwargs and kwargs['remove_duplicates']:
        cleaner.remove_duplicates()
        
    if 'missing_strategy' in kwargs:
        columns = kwargs.get('missing_columns')
        cleaner.handle_missing_values(strategy=kwargs['missing_strategy'], columns=columns)
        
    if 'remove_outliers' in kwargs and kwargs['remove_outliers']:
        columns = kwargs.get('outlier_columns')
        multiplier = kwargs.get('outlier_multiplier', 1.5)
        cleaner.remove_outliers_iqr(columns=columns, multiplier=multiplier)
        
    return cleaner.get_cleaned_data()
import pandas as pd
import re

def clean_dataframe(df, column_names):
    """
    Clean a pandas DataFrame by removing duplicate rows and normalizing
    specified string columns (strip whitespace, lowercase).
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    
    # Normalize specified string columns
    for col in column_names:
        if col in df_cleaned.columns:
            # Convert to string, strip whitespace, convert to lowercase
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip().str.lower()
    
    return df_cleaned

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column using regex.
    Returns a DataFrame with an additional validation column.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['email_valid'] = df[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x)))
    )
    
    return df

def remove_special_characters(df, column_names, keep_chars=''):
    """
    Remove special characters from specified columns, keeping only alphanumeric
    characters and specified additional characters.
    """
    df_cleaned = df.copy()
    
    for col in column_names:
        if col in df_cleaned.columns:
            # Create regex pattern to keep alphanumeric and specified characters
            pattern = f'[^a-zA-Z0-9{re.escape(keep_chars)}]'
            df_cleaned[col] = df_cleaned[col].astype(str).apply(
                lambda x: re.sub(pattern, '', x)
            )
    
    return df_cleaned

def main():
    # Example usage
    sample_data = {
        'name': ['  John Doe  ', 'Jane Smith', 'John Doe', 'Bob@Example'],
        'email': ['john@example.com', 'invalid-email', 'JOHN@EXAMPLE.COM', 'bob@test.org'],
        'phone': ['123-456-7890', '(987) 654-3210', '123-456-7890', '555-1234']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned_df = clean_dataframe(df, ['name', 'email'])
    print("After cleaning:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    # Validate emails
    validated_df = validate_email_column(cleaned_df, 'email')
    print("After email validation:")
    print(validated_df)
    print("\n" + "="*50 + "\n")
    
    # Remove special characters from phone numbers
    final_df = remove_special_characters(validated_df, ['phone'], keep_chars='-')
    print("After removing special characters from phone:")
    print(final_df)

if __name__ == "__main__":
    main()