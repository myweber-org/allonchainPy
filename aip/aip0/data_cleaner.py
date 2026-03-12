
import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = self.df.select_dtypes(exclude=[np.number]).columns.tolist()

    def handle_missing_values(self, strategy='mean', fill_value=None):
        if strategy == 'mean' and self.numeric_columns:
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(self.df[self.numeric_columns].mean())
        elif strategy == 'median' and self.numeric_columns:
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(self.df[self.numeric_columns].median())
        elif strategy == 'mode' and self.categorical_columns:
            for col in self.categorical_columns:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown')
        elif fill_value is not None:
            self.df = self.df.fillna(fill_value)
        return self

    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            columns = self.numeric_columns
        
        for col in columns:
            if col in self.numeric_columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        return self

    def standardize_data(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        for col in columns:
            if col in self.numeric_columns:
                mean = self.df[col].mean()
                std = self.df[col].std()
                if std > 0:
                    self.df[col] = (self.df[col] - mean) / std
        return self

    def get_cleaned_data(self):
        return self.df

    def summary(self):
        print("Data Summary:")
        print(f"Shape: {self.df.shape}")
        print(f"Missing values: {self.df.isnull().sum().sum()}")
        print(f"Columns: {list(self.df.columns)}")import pandas as pd
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

def clean_dataset(file_path):
    df = pd.read_csv(file_path)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
    
    for col in numeric_columns:
        df = normalize_minmax(df, col)
    
    return df

if __name__ == "__main__":
    cleaned_df = clean_dataset('sample_data.csv')
    cleaned_df.to_csv('cleaned_data.csv', index=False)
    print("Data cleaning completed. Cleaned data saved to 'cleaned_data.csv'")import pandas as pd
import re

def clean_dataframe(df, text_columns=None, drop_duplicates=True, lowercase=True):
    """
    Clean a pandas DataFrame by removing duplicates and standardizing text columns.
    
    Args:
        df: pandas DataFrame to clean
        text_columns: list of column names to standardize (if None, auto-detect string columns)
        drop_duplicates: whether to remove duplicate rows
        lowercase: whether to convert text to lowercase
    
    Returns:
        Cleaned pandas DataFrame
    """
    df_clean = df.copy()
    
    if drop_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - len(df_clean)
        print(f"Removed {removed} duplicate rows")
    
    if text_columns is None:
        text_columns = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    for col in text_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str)
            
            if lowercase:
                df_clean[col] = df_clean[col].str.lower()
            
            df_clean[col] = df_clean[col].apply(lambda x: re.sub(r'\s+', ' ', x.strip()))
    
    return df_clean

def validate_email(email):
    """
    Validate email format using regex.
    
    Args:
        email: string to validate as email
    
    Returns:
        Boolean indicating if email is valid
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email)))

def extract_numeric(text):
    """
    Extract numeric values from text.
    
    Args:
        text: string containing numeric values
    
    Returns:
        List of numeric values found in text
    """
    numbers = re.findall(r'\d+\.?\d*', str(text))
    return [float(num) if '.' in num else int(num) for num in numbers]

if __name__ == "__main__":
    sample_data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', 'ALICE JONES'],
        'email': ['john@example.com', 'jane@test.org', 'invalid-email', 'alice@company.co'],
        'notes': ['  Multiple   spaces  ', 'MixedCase Text', 'normal text', '123 main st']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned_df = clean_dataframe(df)
    print(cleaned_df)
    
    print("\nEmail validation:")
    for email in df['email']:
        print(f"{email}: {validate_email(email)}")
    
    print("\nNumeric extraction from notes:")
    for note in df['notes']:
        print(f"{note}: {extract_numeric(note)}")