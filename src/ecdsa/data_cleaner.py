
import pandas as pd
import numpy as np

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, fill_missing=True):
    """
    Clean a pandas DataFrame by standardizing columns, removing duplicates,
    and handling missing values.
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        for col in cleaned_df.select_dtypes(include=[np.number]).columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[col] = cleaned_df[col].fillna('Unknown')
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

def validate_dataframe(df, required_columns=None, unique_columns=None):
    """
    Validate DataFrame structure and data integrity.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if unique_columns:
        for col in unique_columns:
            if col in df.columns and df[col].duplicated().any():
                raise ValueError(f"Column '{col}' contains duplicate values")
    
    return True

def process_csv_file(input_path, output_path, **kwargs):
    """
    Read CSV file, clean data, and save to output path.
    """
    df = pd.read_csv(input_path)
    cleaned_df = clean_dataframe(df, **kwargs)
    cleaned_df.to_csv(output_path, index=False)
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Alice', None, 'Charlie'],
        'Age': [25, 30, 25, 40, None],
        'City': ['NYC', 'LA', 'NYC', 'Chicago', 'Boston']
    })
    
    cleaned = clean_dataframe(sample_data)
    print("Original DataFrame:")
    print(sample_data)
    print("\nCleaned DataFrame:")
    print(cleaned)
import numpy as np
import pandas as pd

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
    
    return filtered_df.reset_index(drop=True)

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to analyze
    
    Returns:
        dict: Dictionary containing statistics
    """
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': len(df)
    }
    return stats

def clean_dataset(df, numeric_columns):
    """
    Clean dataset by removing outliers from multiple numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        numeric_columns (list): List of column names to clean
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[column]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    sample_data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100),
        'score': np.random.uniform(0, 100, 100)
    }
    
    # Introduce some outliers
    sample_data['value'][10] = 500
    sample_data['value'][20] = -200
    sample_data['score'][30] = 150
    
    df = pd.DataFrame(sample_data)
    print("Original dataset shape:", df.shape)
    print("\nOriginal statistics for 'value':")
    print(calculate_statistics(df, 'value'))
    
    cleaned_df = clean_dataset(df, ['value', 'score'])
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("\nCleaned statistics for 'value':")
    print(calculate_statistics(cleaned_df, 'value'))import pandas as pd
import numpy as np
from typing import List, Optional

def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def normalize_column(df: pd.DataFrame, column: str, method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize specified column using selected method.
    
    Args:
        df: Input DataFrame
        column: Column name to normalize
        method: Normalization method ('minmax' or 'zscore')
    
    Returns:
        DataFrame with normalized column
    """
    df = df.copy()
    
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val > min_val:
            df[column] = (df[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val > 0:
            df[column] = (df[column] - mean_val) / std_val
    
    return df

def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Handle missing values in numeric columns.
    
    Args:
        df: Input DataFrame
        strategy: Imputation strategy ('mean', 'median', or 'drop')
    
    Returns:
        DataFrame with handled missing values
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        return df.dropna(subset=numeric_cols)
    
    for col in numeric_cols:
        if strategy == 'mean':
            fill_value = df[col].mean()
        elif strategy == 'median':
            fill_value = df[col].median()
        else:
            continue
        
        df[col] = df[col].fillna(fill_value)
    
    return df

def clean_dataset(df: pd.DataFrame, 
                  deduplicate: bool = True,
                  normalize_cols: Optional[List[str]] = None,
                  missing_strategy: str = 'mean') -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        deduplicate: Whether to remove duplicates
        normalize_cols: Columns to normalize
        missing_strategy: Strategy for handling missing values
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if deduplicate:
        cleaned_df = remove_duplicates(cleaned_df)
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    if normalize_cols:
        for col in normalize_cols:
            if col in cleaned_df.columns:
                cleaned_df = normalize_column(cleaned_df, col)
    
    return cleaned_df
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a dataset by handling missing values and removing outliers.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'drop')
        outlier_threshold (float): Z-score threshold for outlier detection
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean())
    elif missing_strategy == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median())
    elif missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    
    # Remove outliers using Z-score method
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs((cleaned_df[numeric_cols] - cleaned_df[numeric_cols].mean()) / cleaned_df[numeric_cols].std())
    outlier_mask = (z_scores < outlier_threshold).all(axis=1)
    cleaned_df = cleaned_df[outlier_mask]
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=10):
    """
    Validate dataset structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
        min_rows (int): Minimum number of rows required
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"Dataset has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Dataset is valid"

# Example usage
if __name__ == "__main__":
    # Create sample data with missing values and outliers
    sample_data = {
        'temperature': [22.5, 23.1, np.nan, 24.3, 100.0, 22.8, 23.5, 21.9],
        'humidity': [45, 47, 49, np.nan, 48, 46, 200, 44],
        'pressure': [1013, 1012, 1014, 1011, 1013, 1015, 1012, 1014]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\nDataset info:")
    print(df.info())
    
    # Clean the dataset
    cleaned_df = clean_dataset(df, missing_strategy='mean', outlier_threshold=2.5)
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    # Validate the cleaned dataset
    is_valid, message = validate_data(cleaned_df, required_columns=['temperature', 'humidity', 'pressure'])
    print(f"\nValidation result: {is_valid}")
    print(f"Validation message: {message}")