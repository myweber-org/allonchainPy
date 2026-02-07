import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """Remove duplicate rows from DataFrame."""
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df, strategy='mean', columns=None):
    """Fill missing values using specified strategy."""
    df_filled = df.copy()
    
    if columns is None:
        columns = df.columns
    
    for col in columns:
        if df[col].dtype in ['int64', 'float64']:
            if strategy == 'mean':
                df_filled[col] = df[col].fillna(df[col].mean())
            elif strategy == 'median':
                df_filled[col] = df[col].fillna(df[col].median())
            elif strategy == 'mode':
                df_filled[col] = df[col].fillna(df[col].mode()[0])
        else:
            df_filled[col] = df[col].fillna('Unknown')
    
    return df_filled

def normalize_column(df, column, method='minmax'):
    """Normalize a numeric column using specified method."""
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

def filter_outliers(df, column, method='iqr', threshold=1.5):
    """Filter outliers from a column using specified method."""
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return df

def clean_dataset(df, config):
    """Apply multiple cleaning operations based on configuration."""
    cleaned_df = df.copy()
    
    if config.get('remove_duplicates'):
        cleaned_df = remove_duplicates(cleaned_df, config.get('duplicate_subset'))
    
    if config.get('fill_missing'):
        cleaned_df = fill_missing_values(
            cleaned_df, 
            config.get('fill_strategy', 'mean'),
            config.get('fill_columns')
        )
    
    if config.get('normalize'):
        for col, method in config.get('normalize_columns', {}).items():
            if col in cleaned_df.columns:
                cleaned_df = normalize_column(cleaned_df, col, method)
    
    if config.get('remove_outliers'):
        for col in config.get('outlier_columns', []):
            if col in cleaned_df.columns:
                cleaned_df = filter_outliers(
                    cleaned_df, 
                    col, 
                    config.get('outlier_method', 'iqr'),
                    config.get('outlier_threshold', 1.5)
                )
    
    return cleaned_dfimport numpy as np
import pandas as pd

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
    if max_val - min_val == 0:
        return df[column].apply(lambda x: 0.0)
    return (df[column] - min_val) / (max_val - min_val)

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(df):
    required_checks = [
        (lambda d: isinstance(d, pd.DataFrame), "Input must be a pandas DataFrame"),
        (lambda d: not d.empty, "DataFrame cannot be empty"),
        (lambda d: d.isnull().sum().sum() == 0, "DataFrame contains null values")
    ]
    for check, message in required_checks:
        if not check(df):
            raise ValueError(message)
    return True

if __name__ == "__main__":
    sample_data = {'values': [10, 12, 12, 13, 12, 11, 10, 100, 12, 14, 12, 10]}
    df = pd.DataFrame(sample_data)
    print("Original data:")
    print(df)
    
    try:
        validate_dataframe(df)
        cleaned = clean_dataset(df, ['values'])
        print("\nCleaned data:")
        print(cleaned)
    except ValueError as e:
        print(f"Validation error: {e}")
import pandas as pd
import numpy as np

def clean_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame using specified strategy.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Strategy for handling missing values - 'mean', 'median', 'mode', or 'drop'
    columns (list): Specific columns to apply cleaning to, None for all columns
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.columns
    
    for col in columns:
        if col in df_clean.columns:
            if strategy == 'mean':
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            elif strategy == 'median':
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            elif strategy == 'mode':
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                df_clean.dropna(subset=[col], inplace=True)
    
    return df_clean

def remove_outliers(df, columns=None, threshold=3):
    """
    Remove outliers using z-score method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): Columns to check for outliers
    threshold (float): Z-score threshold for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df_clean.columns and df_clean[col].dtype in [np.float64, np.int64]:
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            df_clean = df_clean[z_scores < threshold]
    
    return df_clean

def standardize_columns(df, columns=None):
    """
    Standardize numeric columns to have zero mean and unit variance.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): Columns to standardize
    
    Returns:
    pd.DataFrame: DataFrame with standardized columns
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df_clean.columns and df_clean[col].dtype in [np.float64, np.int64]:
            mean_val = df_clean[col].mean()
            std_val = df_clean[col].std()
            if std_val > 0:
                df_clean[col] = (df_clean[col] - mean_val) / std_val
    
    return df_clean

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    """Remove outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_column(df, column):
    """Normalize column using min-max scaling."""
    min_val = df[column].min()
    max_val = df[column].max()
    df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_data(df, numeric_columns):
    """Main data cleaning pipeline."""
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_column(cleaned_df, col)
    
    cleaned_df = cleaned_df.dropna()
    return cleaned_df

def save_cleaned_data(df, output_path):
    """Save cleaned data to CSV."""
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    raw_data = load_dataset(input_file)
    numeric_cols = raw_data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = clean_data(raw_data, numeric_cols)
    save_cleaned_data(cleaned_data, output_file)
    
    print(f"Original shape: {raw_data.shape}")
    print(f"Cleaned shape: {cleaned_data.shape}")
    print(f"Data cleaning completed. Saved to {output_file}")import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_strategy='mean'):
    """
    Load and clean a CSV file by handling missing values.
    
    Args:
        filepath (str): Path to the CSV file.
        fill_strategy (str): Strategy for filling missing values.
            Options: 'mean', 'median', 'zero', 'drop'.
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    if fill_strategy == 'drop':
        df_cleaned = df.dropna(subset=numeric_columns)
    else:
        df_cleaned = df.copy()
        for col in numeric_columns:
            if df[col].isnull().any():
                if fill_strategy == 'mean':
                    fill_value = df[col].mean()
                elif fill_strategy == 'median':
                    fill_value = df[col].median()
                elif fill_strategy == 'zero':
                    fill_value = 0
                else:
                    raise ValueError(f"Unsupported fill strategy: {fill_strategy}")
                df_cleaned[col] = df[col].fillna(fill_value)
    
    return df_cleaned

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Args:
        df (pandas.DataFrame): Input DataFrame.
        column (str): Column name to process.
        multiplier (float): IQR multiplier for outlier detection.
    
    Returns:
        pandas.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def standardize_columns(df, columns=None):
    """
    Standardize specified columns to have zero mean and unit variance.
    
    Args:
        df (pandas.DataFrame): Input DataFrame.
        columns (list): List of column names to standardize.
            If None, standardize all numeric columns.
    
    Returns:
        pandas.DataFrame: DataFrame with standardized columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_standardized = df.copy()
    
    for col in columns:
        if col in df.columns and np.issubdtype(df[col].dtype, np.number):
            mean_val = df[col].mean()
            std_val = df[col].std()
            
            if std_val > 0:
                df_standardized[col] = (df[col] - mean_val) / std_val
            else:
                df_standardized[col] = 0
    
    return df_standardized

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, 20, 30, np.nan, 50],
        'C': ['x', 'y', 'z', 'x', 'y']
    }
    
    df_sample = pd.DataFrame(sample_data)
    df_sample.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', fill_strategy='mean')
    print("Cleaned DataFrame:")
    print(cleaned_df)
    
    import os
    os.remove('sample_data.csv')
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
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

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing summary statistics.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100)
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    print("Original statistics:", calculate_summary_statistics(df, 'value'))
    
    cleaned_df = remove_outliers_iqr(df, 'value')
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned statistics:", calculate_summary_statistics(cleaned_df, 'value'))
    
    return cleaned_df

if __name__ == "__main__":
    result_df = example_usage()import pandas as pd
import numpy as np

def clean_dataframe(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    missing_strategy (str): Strategy for handling missing values. 
                            Options: 'mean', 'median', 'drop', 'fill_zero'.
    outlier_threshold (float): Number of standard deviations to consider as outlier.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    
    if missing_strategy == 'mean':
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(
            cleaned_df[numeric_cols].mean()
        )
    elif missing_strategy == 'median':
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(
            cleaned_df[numeric_cols].median()
        )
    elif missing_strategy == 'fill_zero':
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(0)
    elif missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna(subset=numeric_cols)
    
    # Remove outliers using z-score method
    if outlier_threshold > 0:
        z_scores = np.abs((cleaned_df[numeric_cols] - cleaned_df[numeric_cols].mean()) / 
                         cleaned_df[numeric_cols].std())
        outlier_mask = (z_scores < outlier_threshold).all(axis=1)
        cleaned_df = cleaned_df[outlier_mask].reset_index(drop=True)
    
    # Clean column names
    cleaned_df.columns = cleaned_df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

# Example usage
if __name__ == "__main__":
    # Create sample data with missing values and outliers
    sample_data = {
        'Age': [25, 30, np.nan, 35, 150, 28, 32],
        'Salary': [50000, 60000, 70000, np.nan, 80000, 90000, 1000000],
        'Experience': [2, 5, 3, 10, 1, 4, 20]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned = clean_dataframe(df, missing_strategy='median', outlier_threshold=2.5)
    print("Cleaned DataFrame:")
    print(cleaned)
    
    # Validate the cleaned data
    is_valid, message = validate_dataframe(cleaned, required_columns=['age', 'salary'], min_rows=3)
    print(f"\nValidation: {is_valid} - {message}")