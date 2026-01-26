
import pandas as pd
import numpy as np

def remove_outliers(df, column, threshold=3):
    mean = df[column].mean()
    std = df[column].std()
    z_scores = np.abs((df[column] - mean) / std)
    return df[z_scores < threshold]

def normalize_column(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path, output_path):
    df = pd.read_csv(file_path)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers(df, col)
        df = normalize_column(df, col)
    
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    cleaned_df = clean_dataset(input_file, output_file)
    print(f"Data cleaning completed. Saved to {output_file}")
    print(f"Original shape: {pd.read_csv(input_file).shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: pandas DataFrame to clean
        drop_duplicates: Boolean indicating whether to drop duplicate rows
        fill_missing: Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    else:
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            if cleaned_df[column].isnull().any():
                if fill_missing == 'mean':
                    cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
                elif fill_missing == 'median':
                    cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
                elif fill_missing == 'mode':
                    cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
    
    return cleaned_df

def remove_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Remove outliers from specified columns using IQR or Z-score method.
    
    Args:
        df: pandas DataFrame
        columns: List of column names to check for outliers (None for all numeric columns)
        method: 'iqr' for Interquartile Range or 'zscore' for Z-score method
        threshold: Threshold multiplier for IQR or Z-score cutoff
    
    Returns:
        DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    
    for column in columns:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            if method == 'iqr':
                Q1 = df_clean[column].quantile(0.25)
                Q3 = df_clean[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_clean = df_clean[(df_clean[column] >= lower_bound) & 
                                   (df_clean[column] <= upper_bound)]
            elif method == 'zscore':
                z_scores = np.abs((df_clean[column] - df_clean[column].mean()) / 
                                 df_clean[column].std())
                df_clean = df_clean[z_scores < threshold]
    
    return df_clean

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize specified columns in the DataFrame.
    
    Args:
        df: pandas DataFrame
        columns: List of column names to normalize (None for all numeric columns)
        method: 'minmax' for Min-Max scaling or 'standard' for Standardization
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    
    for column in columns:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            if method == 'minmax':
                min_val = df_normalized[column].min()
                max_val = df_normalized[column].max()
                if max_val != min_val:
                    df_normalized[column] = (df_normalized[column] - min_val) / (max_val - min_val)
            elif method == 'standard':
                mean_val = df_normalized[column].mean()
                std_val = df_normalized[column].std()
                if std_val != 0:
                    df_normalized[column] = (df_normalized[column] - mean_val) / std_val
    
    return df_normalized

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"import pandas as pd
import numpy as np
from typing import Optional

def clean_csv_data(
    input_path: str,
    output_path: str,
    missing_strategy: str = 'mean',
    numeric_columns: Optional[list] = None,
    date_columns: Optional[list] = None
) -> pd.DataFrame:
    """
    Clean CSV data by handling missing values and converting data types.
    
    Parameters:
    input_path: Path to input CSV file
    output_path: Path to save cleaned CSV file
    missing_strategy: Strategy for handling missing values ('mean', 'median', 'drop', 'zero')
    numeric_columns: List of column names to treat as numeric
    date_columns: List of column names to parse as dates
    
    Returns:
    Cleaned DataFrame
    """
    
    # Read the CSV file
    df = pd.read_csv(input_path)
    
    # Convert specified columns to numeric
    if numeric_columns:
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert specified columns to datetime
    if date_columns:
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Handle missing values based on strategy
    if missing_strategy == 'mean':
        df = df.fillna(df.mean(numeric_only=True))
    elif missing_strategy == 'median':
        df = df.fillna(df.median(numeric_only=True))
    elif missing_strategy == 'zero':
        df = df.fillna(0)
    elif missing_strategy == 'drop':
        df = df.dropna()
    else:
        raise ValueError(f"Unknown missing strategy: {missing_strategy}")
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Reset index after cleaning
    df = df.reset_index(drop=True)
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    
    return df

def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validate that DataFrame contains all required columns.
    
    Parameters:
    df: DataFrame to validate
    required_columns: List of required column names
    
    Returns:
    Boolean indicating if all required columns are present
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return False
    
    return True

def calculate_statistics(df: pd.DataFrame) -> dict:
    """
    Calculate basic statistics for numeric columns.
    
    Parameters:
    df: DataFrame to analyze
    
    Returns:
    Dictionary containing statistics
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return {}
    
    stats = {
        'mean': numeric_df.mean().to_dict(),
        'median': numeric_df.median().to_dict(),
        'std': numeric_df.std().to_dict(),
        'min': numeric_df.min().to_dict(),
        'max': numeric_df.max().to_dict()
    }
    
    return stats

if __name__ == "__main__":
    # Example usage
    cleaned_df = clean_csv_data(
        input_path='raw_data.csv',
        output_path='cleaned_data.csv',
        missing_strategy='mean',
        numeric_columns=['age', 'salary', 'score'],
        date_columns=['join_date']
    )
    
    if validate_dataframe(cleaned_df, ['id', 'name', 'age']):
        stats = calculate_statistics(cleaned_df)
        print(f"Data statistics: {stats}")
        print(f"Cleaned data shape: {cleaned_df.shape}")
        print(f"Cleaned data saved to: cleaned_data.csv")import csv
import os
from typing import List, Dict, Optional

class DataCleaner:
    def __init__(self, input_file: str, output_file: Optional[str] = None):
        self.input_file = input_file
        self.output_file = output_file or self._generate_output_filename()
        self.data = []
        self.headers = []

    def _generate_output_filename(self) -> str:
        base, ext = os.path.splitext(self.input_file)
        return f"{base}_cleaned{ext}"

    def load_data(self) -> None:
        try:
            with open(self.input_file, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                self.headers = reader.fieldnames or []
                self.data = [row for row in reader]
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file '{self.input_file}' not found.")
        except Exception as e:
            raise RuntimeError(f"Error loading data: {str(e)}")

    def remove_empty_rows(self) -> None:
        self.data = [row for row in self.data if any(row.values())]

    def trim_whitespace(self) -> None:
        for row in self.data:
            for key in row:
                if isinstance(row[key], str):
                    row[key] = row[key].strip()

    def fill_missing_values(self, default: str = "N/A") -> None:
        for row in self.data:
            for key in self.headers:
                if not row.get(key):
                    row[key] = default

    def remove_duplicates(self, subset: Optional[List[str]] = None) -> None:
        if not self.data:
            return
        
        seen = set()
        unique_data = []
        
        for row in self.data:
            if subset:
                key = tuple(row[col] for col in subset)
            else:
                key = tuple(row.values())
            
            if key not in seen:
                seen.add(key)
                unique_data.append(row)
        
        self.data = unique_data

    def save_data(self) -> None:
        if not self.data:
            raise ValueError("No data to save. Please load data first.")
        
        with open(self.output_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=self.headers)
            writer.writeheader()
            writer.writerows(self.data)

    def clean(self, 
              remove_empty: bool = True,
              trim_whitespace: bool = True,
              fill_missing: bool = True,
              default_value: str = "N/A",
              remove_dups: bool = True,
              duplicate_subset: Optional[List[str]] = None) -> str:
        
        self.load_data()
        
        if remove_empty:
            self.remove_empty_rows()
        
        if trim_whitespace:
            self.trim_whitespace()
        
        if fill_missing:
            self.fill_missing_values(default_value)
        
        if remove_dups:
            self.remove_duplicates(duplicate_subset)
        
        self.save_data()
        return self.output_file

def example_usage():
    cleaner = DataCleaner("raw_data.csv")
    output_file = cleaner.clean(
        remove_empty=True,
        trim_whitespace=True,
        fill_missing=True,
        default_value="UNKNOWN",
        remove_dups=True,
        duplicate_subset=["id", "email"]
    )
    print(f"Cleaned data saved to: {output_file}")

if __name__ == "__main__":
    example_usage()
import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Remove outliers using z-score
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    df = df[(z_scores < 3).all(axis=1)]
    
    # Normalize numeric columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())
    
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

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
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
    
    return filtered_df

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): List of column names to clean
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    original_shape = cleaned_df.shape
    
    for col in columns:
        if col in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[col]):
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    removed_count = original_shape[0] - cleaned_df.shape[0]
    print(f"Removed {removed_count} outliers from {len(columns)} columns")
    
    return cleaned_df

def validate_dataframe(df):
    """
    Validate DataFrame structure and data quality.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
    }
    
    return validation_resultsimport pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Method to fill missing values. Options are 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
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
            cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None, inplace=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame by checking for required columns and data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names. Default is None.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    if df.empty:
        print("DataFrame is empty.")
        return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 30, 40, 50],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataframe(df, fill_missing='median')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_dataframe(cleaned, required_columns=['A', 'B'])
    print(f"\nDataFrame validation result: {is_valid}")
import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    """
    Load CSV data, remove outliers, and normalize numerical columns.
    """
    df = pd.read_csv(filepath)
    
    # Identify numerical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Remove outliers using z-score method
    z_scores = np.abs(stats.zscore(df[numeric_cols].fillna(0)))
    outlier_mask = (z_scores < 3).all(axis=1)
    df_clean = df[outlier_mask].copy()
    
    # Normalize numerical columns
    for col in numeric_cols:
        if df_clean[col].std() > 0:
            df_clean[col] = (df_clean[col] - df_clean[col].mean()) / df_clean[col].std()
    
    return df_clean

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to CSV.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = load_and_clean_data(input_file)
    save_cleaned_data(cleaned_df, output_file)import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns=None, factor=1.5):
    """
    Remove outliers using Interquartile Range method.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to process (default: all numeric columns)
        factor: IQR multiplier for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_minmax(df, columns=None, feature_range=(0, 1)):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to normalize (default: all numeric columns)
        feature_range: tuple of (min, max) for scaled range
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_norm = df.copy()
    min_val, max_val = feature_range
    
    for col in columns:
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_max != col_min:
                df_norm[col] = min_val + (df[col] - col_min) * (max_val - min_val) / (col_max - col_min)
            else:
                df_norm[col] = min_val
    
    return df_norm

def standardize_zscore(df, columns=None):
    """
    Standardize data using Z-score normalization.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to standardize (default: all numeric columns)
    
    Returns:
        DataFrame with standardized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_std = df.copy()
    
    for col in columns:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            
            if std_val > 0:
                df_std[col] = (df[col] - mean_val) / std_val
            else:
                df_std[col] = 0
    
    return df_std

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        df: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'drop'
        columns: list of column names to process (default: all columns)
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = df.columns
    
    df_processed = df.copy()
    
    for col in columns:
        if col in df.columns and df[col].isnull().any():
            if strategy == 'drop':
                df_processed = df_processed.dropna(subset=[col])
            elif strategy == 'mean':
                df_processed[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'median':
                df_processed[col].fillna(df[col].median(), inplace=True)
            elif strategy == 'mode':
                df_processed[col].fillna(df[col].mode()[0], inplace=True)
    
    return df_processed.reset_index(drop=True)