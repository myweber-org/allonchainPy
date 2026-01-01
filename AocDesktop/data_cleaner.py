
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean dataset by handling missing values and removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'drop')
    outlier_threshold (float): Z-score threshold for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    df_clean = df.copy()
    
    # Handle missing values
    if missing_strategy == 'mean':
        df_clean = df_clean.fillna(df_clean.mean())
    elif missing_strategy == 'median':
        df_clean = df_clean.fillna(df_clean.median())
    elif missing_strategy == 'drop':
        df_clean = df_clean.dropna()
    
    # Remove outliers using Z-score method
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
        df_clean = df_clean[z_scores < outlier_threshold]
    
    # Reset index after cleaning
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean

def validate_dataset(df):
    """
    Validate dataset for common data quality issues.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    dict: Dictionary containing validation results
    """
    
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns)
    }
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 9],
        'C': ['x', 'y', 'z', 'x', 'y']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\nValidation results:")
    print(validate_dataset(df))
    
    cleaned_df = clean_dataset(df, missing_strategy='mean', outlier_threshold=2)
    print("\nCleaned dataset:")
    print(cleaned_df)
    print("\nCleaned validation results:")
    print(validate_dataset(cleaned_df))
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
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
    
    if max_val == min_val:
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
    Comprehensive data cleaning function
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, column)
        
        if normalize_method == 'minmax':
            cleaned_data[f'{column}_normalized'] = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data[f'{column}_normalized'] = normalize_zscore(cleaned_data, column)
    
    return cleaned_data

def get_summary_statistics(data, column):
    """
    Get summary statistics for a column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats_dict = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max(),
        'q1': data[column].quantile(0.25),
        'q3': data[column].quantile(0.75),
        'count': data[column].count(),
        'missing': data[column].isnull().sum()
    }
    
    return stats_dictimport pandas as pd

def clean_data(df):
    """
    Remove duplicate rows and fill missing numeric values with column mean.
    """
    # Remove duplicates
    df_cleaned = df.drop_duplicates()

    # Fill missing numeric values
    numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
    df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())

    return df_cleaned

def load_and_clean(filepath):
    """
    Load a CSV file and apply cleaning operations.
    """
    try:
        df = pd.read_csv(filepath)
        cleaned_df = clean_data(df)
        return cleaned_df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"

    cleaned_data = load_and_clean(input_file)
    if cleaned_data is not None:
        cleaned_data.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
import pandas as pd
import numpy as np

def clean_dataframe(df):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing column names.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    
    # Normalize column names: lowercase and replace spaces with underscores
    df_cleaned.columns = df_cleaned.columns.str.lower().str.replace(' ', '_')
    
    # Reset index after cleaning
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    return df_cleaned

def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in numeric columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == 'median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif strategy == 'mode':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mode().iloc[0])
    
    return df

def validate_dataframe(df):
    """
    Perform basic validation on DataFrame.
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if df.isnull().all().any():
        raise ValueError("DataFrame contains columns with all missing values")
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'Name': ['Alice', 'Bob', 'Alice', 'Charlie'],
        'Age': [25, 30, 25, np.nan],
        'Salary': [50000, 60000, 50000, 70000]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataframe(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    filled_df = handle_missing_values(cleaned_df, strategy='mean')
    print("\nDataFrame after handling missing values:")
    print(filled_df)
    
    try:
        validate_dataframe(filled_df)
        print("\nData validation passed")
    except ValueError as e:
        print(f"\nData validation failed: {e}")
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
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

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from multiple columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to clean. If None, uses all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[col]):
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            except Exception as e:
                print(f"Warning: Could not process column '{col}': {e}")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        tuple: (is_valid, message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    sample_data = {
        'id': range(1, 21),
        'value': [10, 12, 11, 15, 9, 100, 13, 14, 8, 12, 
                  11, 16, 10, 13, 9, 200, 14, 12, 11, 10]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print(f"\nOriginal shape: {df.shape}")
    
    cleaned_df = clean_numeric_data(df, columns=['value'])
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print(f"\nCleaned shape: {cleaned_df.shape}")
    
    is_valid, message = validate_dataframe(cleaned_df, required_columns=['id', 'value'])
    print(f"\nValidation: {message}")