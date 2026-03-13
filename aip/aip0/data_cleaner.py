
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
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        column_mapping (dict, optional): Dictionary mapping old column names to new ones
        drop_duplicates (bool): Whether to remove duplicate rows
        normalize_text (bool): Whether to normalize text columns (strip, lower case)
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    
    if normalize_text:
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
    
    return cleaned_df

def remove_special_characters(text, keep_pattern=r'[a-zA-Z0-9\s]'):
    """
    Remove special characters from text, keeping only alphanumeric and spaces by default.
    
    Args:
        text (str): Input text
        keep_pattern (str): Regex pattern of characters to keep
    
    Returns:
        str: Cleaned text
    """
    if pd.isna(text):
        return text
    return re.sub(f'[^{keep_pattern}]', '', str(text))

def validate_email(email):
    """
    Validate email format using regex pattern.
    
    Args:
        email (str): Email address to validate
    
    Returns:
        bool: True if email format is valid
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email))) if pd.notna(email) else False

def clean_numeric_column(series, fill_na=0, clip_outliers=True, lower_quantile=0.01, upper_quantile=0.99):
    """
    Clean numeric column by filling NA values and optionally clipping outliers.
    
    Args:
        series (pd.Series): Numeric series to clean
        fill_na: Value to fill NA entries with
        clip_outliers (bool): Whether to clip outliers based on quantiles
        lower_quantile (float): Lower quantile for clipping
        upper_quantile (float): Upper quantile for clipping
    
    Returns:
        pd.Series: Cleaned numeric series
    """
    cleaned = series.fillna(fill_na)
    
    if clip_outliers and len(cleaned) > 0:
        lower_bound = cleaned.quantile(lower_quantile)
        upper_bound = cleaned.quantile(upper_quantile)
        cleaned = cleaned.clip(lower_bound, upper_bound)
    
    return cleaned

if __name__ == "__main__":
    sample_data = {
        'Name': ['John Doe', 'Jane Smith', 'John Doe', ' Bob Johnson ', 'ALICE@EXAMPLE.COM'],
        'Email': ['john@example.com', 'jane@example.com', 'invalid-email', 'bob@example.com', 'alice@example.com'],
        'Age': [25, 30, 25, 35, None],
        'Score': [85, 92, 85, 150, 88]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataframe(df, drop_duplicates=True, normalize_text=True)
    cleaned['Email'] = cleaned['Email'].apply(lambda x: x if validate_email(x) else None)
    cleaned['Age'] = clean_numeric_column(cleaned['Age'], fill_na=cleaned['Age'].mean())
    cleaned['Score'] = clean_numeric_column(cleaned['Score'], clip_outliers=True)
    
    print("Cleaned DataFrame:")
    print(cleaned)
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if min_val == max_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    normalized = (data[column] - mean_val) / std_val
    return normalized

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Clean dataset by removing outliers and normalizing numeric columns
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column not in data.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, column)
        
        if normalize_method == 'minmax':
            cleaned_data[column] = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data[column] = normalize_zscore(cleaned_data, column)
    
    return cleaned_data

def validate_data(data, required_columns=None, check_missing=True, check_duplicates=True):
    """
    Validate dataset for common data quality issues
    """
    validation_report = {}
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            validation_report['missing_columns'] = missing_columns
    
    if check_missing:
        missing_values = data.isnull().sum()
        missing_percentage = (missing_values / len(data)) * 100
        validation_report['missing_values'] = missing_percentage[missing_percentage > 0].to_dict()
    
    if check_duplicates:
        duplicate_count = data.duplicated().sum()
        validation_report['duplicate_rows'] = duplicate_count
    
    return validation_report
import pandas as pd
import numpy as np

def clean_dataframe(df):
    """
    Clean a pandas DataFrame by removing duplicate rows,
    standardizing column names, and filling missing values.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()

    # Standardize column names: lowercase and replace spaces with underscores
    df_cleaned.columns = df_cleaned.columns.str.lower().str.replace(' ', '_')

    # Fill missing numeric values with column median
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

    # Fill missing categorical values with mode
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        mode_value = df_cleaned[col].mode()
        if not mode_value.empty:
            df_cleaned[col] = df_cleaned[col].fillna(mode_value.iloc[0])

    return df_cleaned

def validate_dataframe(df):
    """
    Validate DataFrame for required columns and data integrity.
    """
    required_columns = ['id', 'name', 'value']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for negative values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if (df[col] < 0).any():
            print(f"Warning: Negative values found in column '{col}'")
    
    return True

def process_data(file_path):
    """
    Main function to load, clean, and validate data from a CSV file.
    """
    try:
        # Load data
        df = pd.read_csv(file_path)
        
        # Clean data
        df_cleaned = clean_dataframe(df)
        
        # Validate data
        validate_dataframe(df_cleaned)
        
        # Save cleaned data
        output_path = file_path.replace('.csv', '_cleaned.csv')
        df_cleaned.to_csv(output_path, index=False)
        
        print(f"Data cleaning complete. Cleaned data saved to: {output_path}")
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: File not found at path: {file_path}")
    except Exception as e:
        print(f"Error during data processing: {str(e)}")

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'ID': [1, 2, 2, 3, 4],
        'Name': ['Alice', 'Bob', 'Bob', 'Charlie', None],
        'Value': [100, 200, 200, None, 400],
        'Category': ['A', 'B', 'B', 'A', 'A']
    }
    
    df_sample = pd.DataFrame(sample_data)
    cleaned_df = clean_dataframe(df_sample)
    print("Original DataFrame:")
    print(df_sample)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
import pandas as pd
import numpy as np
from typing import Optional, Union, List

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> 'DataCleaner':
        self.df = self.df.drop_duplicates(subset=subset, keep='first')
        return self
        
    def handle_missing_values(self, 
                             strategy: str = 'mean', 
                             columns: Optional[List[str]] = None) -> 'DataCleaner':
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
        for col in columns:
            if col in self.df.columns:
                if strategy == 'mean':
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif strategy == 'median':
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                elif strategy == 'mode':
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                elif strategy == 'drop':
                    self.df = self.df.dropna(subset=[col])
                elif strategy == 'zero':
                    self.df[col].fillna(0, inplace=True)
                    
        return self
        
    def remove_outliers(self, 
                       method: str = 'iqr', 
                       columns: Optional[List[str]] = None,
                       threshold: float = 1.5) -> 'DataCleaner':
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
        for col in columns:
            if col in self.df.columns:
                if method == 'iqr':
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    self.df = self.df[(self.df[col] >= lower_bound) & 
                                     (self.df[col] <= upper_bound)]
                    
        return self
        
    def normalize_data(self, 
                      method: str = 'minmax',
                      columns: Optional[List[str]] = None) -> 'DataCleaner':
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
        for col in columns:
            if col in self.df.columns:
                if method == 'minmax':
                    min_val = self.df[col].min()
                    max_val = self.df[col].max()
                    if max_val != min_val:
                        self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
                elif method == 'zscore':
                    mean_val = self.df[col].mean()
                    std_val = self.df[col].std()
                    if std_val != 0:
                        self.df[col] = (self.df[col] - mean_val) / std_val
                        
        return self
        
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df.copy()
        
    def get_cleaning_report(self) -> dict:
        final_shape = self.df.shape
        rows_removed = self.original_shape[0] - final_shape[0]
        cols_removed = self.original_shape[1] - final_shape[1]
        
        return {
            'original_shape': self.original_shape,
            'final_shape': final_shape,
            'rows_removed': rows_removed,
            'columns_removed': cols_removed,
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict()
        }
        
    def save_cleaned_data(self, filepath: str, format: str = 'csv'):
        if format == 'csv':
            self.df.to_csv(filepath, index=False)
        elif format == 'excel':
            self.df.to_excel(filepath, index=False)
        elif format == 'parquet':
            self.df.to_parquet(filepath, index=False)

def load_and_clean_csv(filepath: str, 
                      missing_strategy: str = 'mean',
                      remove_outliers: bool = True) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    cleaner = DataCleaner(df)
    
    cleaner.handle_missing_values(strategy=missing_strategy)
    
    if remove_outliers:
        cleaner.remove_outliers()
        
    cleaner.remove_duplicates()
    
    return cleaner.get_cleaned_data()