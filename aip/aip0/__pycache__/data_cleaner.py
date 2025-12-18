import numpy as np
import pandas as pd

def remove_outliers_iqr(dataframe, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Args:
        dataframe: pandas DataFrame containing the data
        column: string name of the column to process
    
    Returns:
        pandas DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]
    
    return filtered_df

def calculate_basic_stats(dataframe, column):
    """
    Calculate basic statistics for a specified column.
    
    Args:
        dataframe: pandas DataFrame containing the data
        column: string name of the column to analyze
    
    Returns:
        dictionary containing statistical measures
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': dataframe[column].mean(),
        'median': dataframe[column].median(),
        'std': dataframe[column].std(),
        'min': dataframe[column].min(),
        'max': dataframe[column].max(),
        'count': dataframe[column].count()
    }
    
    return stats

def clean_dataset(dataframe, numeric_columns=None):
    """
    Clean dataset by removing outliers from all numeric columns.
    
    Args:
        dataframe: pandas DataFrame to clean
        numeric_columns: list of column names to process (defaults to all numeric columns)
    
    Returns:
        cleaned pandas DataFrame
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = dataframe.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
            except Exception as e:
                print(f"Warning: Could not process column '{column}': {e}")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 27, 28, 29, 30, 100],
        'humidity': [45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
        'pressure': [1013, 1012, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1100]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\nOriginal statistics for 'temperature':")
    print(calculate_basic_stats(df, 'temperature'))
    
    cleaned_df = clean_dataset(df)
    print("\nCleaned dataset:")
    print(cleaned_df)
    print("\nCleaned statistics for 'temperature':")
    print(calculate_basic_stats(cleaned_df, 'temperature'))import pandas as pd

def clean_dataset(df, drop_duplicates=True, fillna_method=None, fillna_value=None):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default True.
    fillna_method (str): Method to fill missing values ('ffill', 'bfill', etc). Default None.
    fillna_value: Value to fill missing values with if fillna_method is None.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fillna_method is not None:
        cleaned_df = cleaned_df.fillna(method=fillna_method)
    elif fillna_value is not None:
        cleaned_df = cleaned_df.fillna(fillna_value)
    
    # Remove duplicates
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate dataset for required columns and basic integrity.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    return True, "Dataset is valid"

# Example usage
if __name__ == "__main__":
    # Create sample data with missing values and duplicates
    sample_data = {
        'id': [1, 2, 2, 3, 4, 4],
        'name': ['Alice', 'Bob', 'Bob', None, 'Eve', 'Eve'],
        'score': [85, 90, 90, 78, None, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nDataFrame info:")
    print(df.info())
    
    # Clean the dataset
    cleaned = clean_dataset(df, fillna_value=0)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    # Validate the cleaned dataset
    is_valid, message = validate_dataset(cleaned, required_columns=['id', 'name', 'score'])
    print(f"\nValidation: {message}")
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (DataFrame): The input data.
    column (str): The column name to clean.
    
    Returns:
    DataFrame: Data with outliers removed.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    return filtered_data

def calculate_summary_statistics(data, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    data (DataFrame): The input data.
    column (str): The column name.
    
    Returns:
    dict: Summary statistics.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    stats = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max()
    }
    
    return statsimport pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to drop duplicate rows
        fill_missing (str): Strategy for filling missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values")
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        print(f"Filled missing numeric values with {fill_missing}")
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown')
        print("Filled missing categorical values with mode")
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for basic quality checks.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        dict: Dictionary with validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_columns
        validation_results['all_required_columns_present'] = len(missing_columns) == 0
    
    return validation_results

def normalize_numeric_columns(df, columns=None, method='minmax'):
    """
    Normalize numeric columns in a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to normalize (defaults to all numeric columns)
        method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
        pd.DataFrame: DataFrame with normalized columns
    """
    normalized_df = df.copy()
    
    if columns is None:
        columns = normalized_df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in normalized_df.columns and pd.api.types.is_numeric_dtype(normalized_df[col]):
            if method == 'minmax':
                col_min = normalized_df[col].min()
                col_max = normalized_df[col].max()
                if col_max > col_min:
                    normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
            elif method == 'zscore':
                col_mean = normalized_df[col].mean()
                col_std = normalized_df[col].std()
                if col_std > 0:
                    normalized_df[col] = (normalized_df[col] - col_mean) / col_std
    
    return normalized_df

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, None, 40.1, 50.0],
        'category': ['A', 'B', 'B', 'C', None, 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\nValidation results:")
    print(validate_dataset(df))
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned dataset:")
    print(cleaned)
    
    normalized = normalize_numeric_columns(cleaned, method='minmax')
    print("\nNormalized dataset:")
    print(normalized)import pandas as pd
import numpy as np

def clean_dataframe(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    missing_strategy (str): Strategy for handling missing values.
        Options: 'mean', 'median', 'mode', 'drop'.
    outlier_threshold (float): Number of standard deviations to consider as outlier.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    
    if missing_strategy == 'mean':
        for col in numeric_cols:
            cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
    elif missing_strategy == 'median':
        for col in numeric_cols:
            cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    elif missing_strategy == 'mode':
        for col in numeric_cols:
            cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
    elif missing_strategy == 'drop':
        cleaned_df.dropna(inplace=True)
    
    # Handle outliers using z-score method
    for col in numeric_cols:
        z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
        cleaned_df = cleaned_df[z_scores < outlier_threshold]
    
    # Reset index after outlier removal
    cleaned_df.reset_index(drop=True, inplace=True)
    
    return cleaned_df

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    subset (list): Columns to consider for identifying duplicates.
    keep (str): Which duplicates to keep.
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed.
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def normalize_columns(df, columns=None, method='minmax'):
    """
    Normalize specified columns in DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): Columns to normalize. If None, normalize all numeric columns.
    method (str): Normalization method. Options: 'minmax', 'zscore'.
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns.
    """
    normalized_df = df.copy()
    
    if columns is None:
        columns = normalized_df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in normalized_df.columns and pd.api.types.is_numeric_dtype(normalized_df[col]):
            if method == 'minmax':
                col_min = normalized_df[col].min()
                col_max = normalized_df[col].max()
                if col_max != col_min:
                    normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
            elif method == 'zscore':
                col_mean = normalized_df[col].mean()
                col_std = normalized_df[col].std()
                if col_std != 0:
                    normalized_df[col] = (normalized_df[col] - col_mean) / col_std
    
    return normalized_df

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, 20, 30, np.nan, 50, 60],
        'C': ['x', 'y', 'x', 'y', 'x', 'y']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    # Clean the data
    cleaned = clean_dataframe(df, missing_strategy='mean', outlier_threshold=2)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    # Normalize columns
    normalized = normalize_columns(cleaned, method='minmax')
    print("\nNormalized DataFrame:")
    print(normalized)