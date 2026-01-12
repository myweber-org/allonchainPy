
import re

def clean_text(text):
    """
    Clean and normalize a given text string.
    
    Args:
        text (str): The input text to be cleaned.
    
    Returns:
        str: The cleaned text with extra whitespace removed and converted to lowercase.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    
    # Remove leading and trailing whitespace
    text = text.strip()
    
    # Replace multiple spaces or newlines with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    return text

def clean_text_list(text_list):
    """
    Clean a list of text strings.
    
    Args:
        text_list (list): A list of text strings to be cleaned.
    
    Returns:
        list: A list of cleaned text strings.
    """
    if not isinstance(text_list, list):
        raise TypeError("Input must be a list")
    
    return [clean_text(text) for text in text_list]
import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None, fill_missing='mean', remove_duplicates=True):
    """
    Clean a pandas DataFrame by handling missing values and removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_check (list): List of columns to check for missing values
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', 'drop')
    remove_duplicates (bool): Whether to remove duplicate rows
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    df_clean = df.copy()
    
    if columns_to_check is None:
        columns_to_check = df_clean.columns.tolist()
    
    if remove_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - len(df_clean)
        print(f"Removed {removed} duplicate rows")
    
    missing_counts = df_clean[columns_to_check].isnull().sum()
    if missing_counts.sum() > 0:
        print("Missing values found:")
        for col, count in missing_counts.items():
            if count > 0:
                print(f"  {col}: {count} missing values")
        
        if fill_missing == 'drop':
            df_clean = df_clean.dropna(subset=columns_to_check)
            print("Rows with missing values dropped")
        else:
            for col in columns_to_check:
                if df_clean[col].isnull().sum() > 0:
                    if fill_missing == 'mean' and pd.api.types.is_numeric_dtype(df_clean[col]):
                        fill_value = df_clean[col].mean()
                    elif fill_missing == 'median' and pd.api.types.is_numeric_dtype(df_clean[col]):
                        fill_value = df_clean[col].median()
                    elif fill_missing == 'mode':
                        fill_value = df_clean[col].mode()[0]
                    else:
                        fill_value = 0 if pd.api.types.is_numeric_dtype(df_clean[col]) else 'Unknown'
                    
                    df_clean[col] = df_clean[col].fillna(fill_value)
                    print(f"Filled missing values in {col} with {fill_value}")
    
    return df_clean

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has less than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return True

def remove_outliers_iqr(df, columns, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of columns to check for outliers
    multiplier (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    
    df_clean = df.copy()
    
    for col in columns:
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            initial_count = len(df_clean)
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            removed = initial_count - len(df_clean)
            
            if removed > 0:
                print(f"Removed {removed} outliers from column '{col}'")
    
    return df_clean

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5, 6],
        'name': ['Alice', 'Bob', 'Charlie', None, 'Eve', 'Eve', 'Frank'],
        'age': [25, 30, None, 40, 35, 35, 150],
        'score': [85.5, 92.0, 78.5, None, 88.0, 88.0, 95.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df, fill_missing='mean', remove_duplicates=True)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    is_valid = validate_dataframe(cleaned_df, required_columns=['id', 'name', 'age', 'score'])
    print(f"\nDataFrame validation: {is_valid}")
    
    df_no_outliers = remove_outliers_iqr(cleaned_df, ['age', 'score'])
    print("\nDataFrame after outlier removal:")
    print(df_no_outliers)import pandas as pd
import numpy as np

def clean_csv_data(input_path, output_path):
    """
    Load a CSV file, clean the data by handling missing values,
    converting data types, and save the cleaned version.
    """
    try:
        df = pd.read_csv(input_path)
        
        # Fill missing numeric values with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        # Convert date columns if present
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Reset index after cleaning
        df = df.reset_index(drop=True)
        
        # Save cleaned data
        df.to_csv(output_path, index=False)
        print(f"Data cleaning completed. Cleaned data saved to: {output_path}")
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df):
    """
    Validate the cleaned dataframe for basic quality checks.
    """
    if df is None or df.empty:
        print("DataFrame is empty or None")
        return False
    
    checks = {
        'has_nulls': df.isnull().sum().sum() == 0,
        'has_duplicates': not df.duplicated().any(),
        'has_rows': len(df) > 0,
        'has_columns': len(df.columns) > 0
    }
    
    all_passed = all(checks.values())
    
    if all_passed:
        print("All data validation checks passed")
    else:
        failed = [k for k, v in checks.items() if not v]
        print(f"Validation failed for: {', '.join(failed)}")
    
    return all_passed

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_file, output_file)
    
    if cleaned_df is not None:
        validate_dataframe(cleaned_df)
import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    for column in df.select_dtypes(include=[np.number]).columns:
        df[column] = df[column].fillna(df[column].median())
    
    # Remove outliers using z-score
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    df = df[(z_scores < 3).all(axis=1)]
    
    # Normalize numeric columns
    for column in numeric_cols:
        if df[column].std() > 0:
            df[column] = (df[column] - df[column].mean()) / df[column].std()
    
    return df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = load_and_clean_data(input_file)
    save_cleaned_data(cleaned_df, output_file)
import pandas as pd
import numpy as np
from pathlib import Path

class DataCleaner:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.df = None
        
    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Loaded {len(self.df)} rows from {self.file_path.name}")
            return True
        except FileNotFoundError:
            print(f"Error: File {self.file_path} not found")
            return False
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def remove_duplicates(self):
        if self.df is None:
            print("No data loaded")
            return 0
            
        initial_count = len(self.df)
        self.df.drop_duplicates(inplace=True)
        removed = initial_count - len(self.df)
        print(f"Removed {removed} duplicate rows")
        return removed
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if self.df is None:
            print("No data loaded")
            return
            
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns:
                missing_count = self.df[col].isnull().sum()
                if missing_count > 0:
                    if strategy == 'mean':
                        fill_value = self.df[col].mean()
                    elif strategy == 'median':
                        fill_value = self.df[col].median()
                    elif strategy == 'mode':
                        fill_value = self.df[col].mode()[0]
                    elif strategy == 'drop':
                        self.df = self.df.dropna(subset=[col])
                        print(f"Dropped {missing_count} rows with missing values in {col}")
                        continue
                    else:
                        fill_value = 0
                    
                    self.df[col].fillna(fill_value, inplace=True)
                    print(f"Filled {missing_count} missing values in {col} with {strategy}: {fill_value}")
    
    def normalize_numeric_columns(self, columns=None):
        if self.df is None:
            print("No data loaded")
            return
            
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in self.df.columns:
                if self.df[col].std() != 0:
                    self.df[col] = (self.df[col] - self.df[col].mean()) / self.df[col].std()
                    print(f"Normalized column: {col}")
    
    def save_cleaned_data(self, output_path=None):
        if self.df is None:
            print("No data to save")
            return False
            
        if output_path is None:
            output_path = self.file_path.parent / f"cleaned_{self.file_path.name}"
        
        try:
            self.df.to_csv(output_path, index=False)
            print(f"Saved cleaned data to {output_path}")
            return True
        except Exception as e:
            print(f"Error saving file: {e}")
            return False
    
    def get_summary(self):
        if self.df is None:
            return "No data loaded"
        
        summary = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns),
            'missing_values': self.df.isnull().sum().sum()
        }
        return summary

def clean_csv_file(input_file, output_file=None):
    cleaner = DataCleaner(input_file)
    
    if not cleaner.load_data():
        return None
    
    print("Starting data cleaning process...")
    cleaner.remove_duplicates()
    cleaner.handle_missing_values(strategy='mean')
    cleaner.normalize_numeric_columns()
    
    if output_file:
        cleaner.save_cleaned_data(output_file)
    else:
        cleaner.save_cleaned_data()
    
    summary = cleaner.get_summary()
    print(f"Cleaning complete. Summary: {summary}")
    
    return cleaner.df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10.5, 20.3, np.nan, 15.7, 30.1, 30.1],
        'category': ['A', 'B', 'A', 'C', 'B', 'B'],
        'score': [85, 92, 78, np.nan, 88, 88]
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    cleaned_df = clean_csv_file('sample_data.csv', 'cleaned_sample.csv')
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to drop duplicate rows
        fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
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
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of column names that must be present
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Dataset is valid"import pandas as pd
import numpy as np

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val == min_val:
        return data[column]
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    mean_val = data[column].mean()
    std_val = data[column].std()
    if std_val == 0:
        return data[column]
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_mean(data, column):
    mean_val = data[column].mean()
    filled_data = data[column].fillna(mean_val)
    return filled_data

def handle_missing_median(data, column):
    median_val = data[column].median()
    filled_data = data[column].fillna(median_val)
    return filled_data

def clean_dataset(data, numeric_columns):
    cleaned_data = data.copy()
    for col in numeric_columns:
        if col in cleaned_data.columns:
            cleaned_data = remove_outliers_iqr(cleaned_data, col)
            cleaned_data[col] = handle_missing_mean(cleaned_data, col)
            cleaned_data[col] = normalize_minmax(cleaned_data, col)
    return cleaned_data

def summarize_cleaning(data, cleaned_data, numeric_columns):
    summary = {}
    for col in numeric_columns:
        if col in data.columns:
            original_count = len(data)
            cleaned_count = len(cleaned_data)
            removed_count = original_count - cleaned_count
            missing_original = data[col].isna().sum()
            missing_cleaned = cleaned_data[col].isna().sum()
            summary[col] = {
                'original_samples': original_count,
                'cleaned_samples': cleaned_count,
                'outliers_removed': removed_count,
                'missing_filled': missing_original - missing_cleaned
            }
    return summary