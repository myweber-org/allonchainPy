import pandas as pd
import numpy as np

def clean_dataframe(df):
    """
    Clean a pandas DataFrame by removing null values and standardizing column names.
    """
    # Remove rows with any null values
    df_cleaned = df.dropna()
    
    # Standardize column names: lowercase and replace spaces with underscores
    df_cleaned.columns = df_cleaned.columns.str.lower().str.replace(' ', '_')
    
    # Reset index after dropping rows
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    return df_cleaned

def validate_dataframe(df):
    """
    Validate that the DataFrame has no null values and column names are standardized.
    """
    # Check for null values
    if df.isnull().sum().sum() > 0:
        raise ValueError("DataFrame contains null values after cleaning.")
    
    # Check column name format
    for col in df.columns:
        if not col.islower() or ' ' in col:
            raise ValueError(f"Column name '{col}' is not properly standardized.")
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'First Name': ['Alice', 'Bob', None, 'David'],
        'Last Name': ['Smith', 'Johnson', 'Williams', None],
        'Age': [25, 30, 35, 40],
        'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned_df = clean_dataframe(df)
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n")
    
    try:
        validate_dataframe(cleaned_df)
        print("Data validation passed.")
    except ValueError as e:
        print(f"Data validation failed: {e}")
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
                
        self.df = df_clean.reset_index(drop=True)
        return self
        
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_normalized = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    df_normalized[col] = (self.df[col] - min_val) / (max_val - min_val)
                    
        self.df = df_normalized
        return self
        
    def standardize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_standardized = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    df_standardized[col] = (self.df[col] - mean_val) / std_val
                    
        self.df = df_standardized
        return self
        
    def handle_missing_mean(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_filled = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                mean_val = self.df[col].mean()
                df_filled[col] = self.df[col].fillna(mean_val)
                
        self.df = df_filled
        return self
        
    def get_cleaned_data(self):
        return self.df
        
    def get_removed_count(self):
        return self.original_shape[0] - self.df.shape[0]import csv
import re

def clean_csv(input_file, output_file):
    """
    Clean a CSV file by removing rows with missing values
    and standardizing text fields.
    """
    cleaned_rows = []
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        for row in reader:
            # Skip rows with any empty values
            if any(value.strip() == '' for value in row.values()):
                continue
            
            # Clean text fields
            cleaned_row = {}
            for key, value in row.items():
                if key.endswith('_name') or key.endswith('_text'):
                    # Remove extra whitespace and standardize case
                    cleaned_value = re.sub(r'\s+', ' ', value.strip())
                    cleaned_value = cleaned_value.title()
                    cleaned_row[key] = cleaned_value
                else:
                    cleaned_row[key] = value.strip()
            
            cleaned_rows.append(cleaned_row)
    
    # Write cleaned data to output file
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)
    
    return len(cleaned_rows)

def validate_email(email):
    """
    Validate email format using regex pattern.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def remove_duplicates(input_file, output_file, key_field):
    """
    Remove duplicate rows based on a specified key field.
    """
    seen = set()
    unique_rows = []
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        for row in reader:
            key = row.get(key_field)
            if key not in seen:
                seen.add(key)
                unique_rows.append(row)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unique_rows)
    
    return len(unique_rows)

if __name__ == "__main__":
    # Example usage
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    try:
        rows_processed = clean_csv(input_csv, output_csv)
        print(f"Processed {rows_processed} rows successfully.")
        
        # Remove duplicates based on 'id' field
        deduplicated_csv = "deduplicated_data.csv"
        unique_count = remove_duplicates(output_csv, deduplicated_csv, 'id')
        print(f"Found {unique_count} unique records.")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_csv}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a specified column in a DataFrame using the IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to process.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
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

def clean_missing_values(df, strategy='mean'):
    """
    Handle missing values in numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        strategy (str): Imputation strategy ('mean', 'median', 'mode', 'drop').
    
    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        return df.dropna(subset=numeric_cols)
    
    for col in numeric_cols:
        if df[col].isnull().any():
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0]
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            df[col] = df[col].fillna(fill_value)
    
    return df

def normalize_column(df, column):
    """
    Normalize a column using min-max scaling.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to normalize.
    
    Returns:
        pd.DataFrame: DataFrame with normalized column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = df[column].min()
    max_val = df[column].max()
    
    if max_val == min_val:
        df[column + '_normalized'] = 0.5
    else:
        df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    
    return df

def get_data_summary(df):
    """
    Generate a summary statistics DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
    
    Returns:
        pd.DataFrame: Summary statistics.
    """
    summary = pd.DataFrame({
        'dtype': df.dtypes,
        'non_null_count': df.count(),
        'null_count': df.isnull().sum(),
        'null_percentage': (df.isnull().sum() / len(df)) * 100
    })
    
    numeric_summary = df.describe().transpose()
    summary = summary.join(numeric_summary, how='left')
    
    return summary