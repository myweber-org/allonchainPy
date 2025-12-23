import pandas as pd
import sys

def remove_duplicates(input_file, output_file=None, subset=None):
    """
    Remove duplicate rows from a CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str, optional): Path to output CSV file. If None, overwrites input file
        subset (list, optional): List of column names to consider for identifying duplicates
    """
    try:
        df = pd.read_csv(input_file)
        
        if subset:
            df_clean = df.drop_duplicates(subset=subset, keep='first')
        else:
            df_clean = df.drop_duplicates(keep='first')
        
        if output_file is None:
            output_file = input_file
        
        df_clean.to_csv(output_file, index=False)
        print(f"Removed {len(df) - len(df_clean)} duplicate rows")
        print(f"Cleaned data saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: File '{input_file}' is empty")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_file> [output_file]")
        print("Example: python data_cleaner.py data.csv cleaned_data.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    remove_duplicates(input_file, output_file)import pandas as pd

def clean_dataset(df, columns_to_check=None, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicate rows and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_check (list, optional): List of column names to check for duplicates.
                                          If None, checks all columns. Defaults to None.
        fill_missing (str, optional): Strategy to fill missing values.
                                     Options: 'mean', 'median', 'mode', or 'drop'.
                                     Defaults to 'mean'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    if columns_to_check:
        cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check)
    else:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Handle missing values
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
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
            else:
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of required column names.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = {
#         'A': [1, 2, 2, None, 5],
#         'B': [10, 20, 20, 40, None],
#         'C': ['x', 'y', 'y', None, 'z']
#     }
#     df = pd.DataFrame(sample_data)
#     cleaned = clean_dataset(df, fill_missing='median')
#     print("Original shape:", df.shape)
#     print("Cleaned shape:", cleaned.shape)
#     print("Validation:", validate_dataframe(cleaned))import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, handle_nulls='drop', fill_value=None):
    """
    Clean a pandas DataFrame by handling duplicates and null values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    handle_nulls (str): Strategy for null handling - 'drop', 'fill', or 'ignore'.
    fill_value: Value to fill nulls with if handle_nulls is 'fill'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if handle_nulls == 'drop':
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.dropna()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} rows with null values")
    elif handle_nulls == 'fill':
        if fill_value is not None:
            cleaned_df = cleaned_df.fillna(fill_value)
            print(f"Filled null values with {fill_value}")
        else:
            cleaned_df = cleaned_df.fillna(0)
            print("Filled null values with 0")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    return True

def calculate_missing_percentage(df):
    """
    Calculate percentage of missing values for each column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    
    Returns:
    pd.Series: Percentage of missing values per column.
    """
    if df.empty:
        return pd.Series()
    
    total_rows = len(df)
    missing_counts = df.isnull().sum()
    missing_percentage = (missing_counts / total_rows) * 100
    
    return missing_percentage.round(2)

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', None, 'Eve', None],
        'score': [85, 92, 92, 78, None, 88],
        'age': [25, 30, 30, 22, 28, 35]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nMissing values percentage:")
    print(calculate_missing_percentage(df))
    
    cleaned = clean_dataset(df, handle_nulls='fill', fill_value='Unknown')
    print("\nCleaned DataFrame:")
    print(cleaned)import pandas as pd

def clean_dataset(df, remove_duplicates=True, fill_method='drop'):
    """
    Clean a pandas DataFrame by handling missing values and optionally removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    remove_duplicates (bool): If True, remove duplicate rows.
    fill_method (str): Method to handle missing values: 'drop', 'fill_mean', 'fill_median', 'fill_mode'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_method == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_method == 'fill_mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
    elif fill_method == 'fill_median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
    elif fill_method == 'fill_mode':
        cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
    else:
        raise ValueError("Invalid fill_method. Choose from 'drop', 'fill_mean', 'fill_median', 'fill_mode'.")
    
    # Remove duplicates if requested
    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    dict: Dictionary with validation results.
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'summary': {}
    }
    
    # Check if input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        validation_result['is_valid'] = False
        validation_result['errors'].append('Input is not a pandas DataFrame')
        return validation_result
    
    # Check for required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f'Missing required columns: {missing_columns}')
    
    # Calculate basic statistics
    validation_result['summary']['total_rows'] = len(df)
    validation_result['summary']['total_columns'] = len(df.columns)
    validation_result['summary']['null_count'] = df.isnull().sum().sum()
    validation_result['summary']['duplicate_rows'] = df.duplicated().sum()
    
    # Add warnings for potential issues
    if validation_result['summary']['null_count'] > 0:
        validation_result['warnings'].append(f'Dataset contains {validation_result["summary"]["null_count"]} null values')
    
    if validation_result['summary']['duplicate_rows'] > 0:
        validation_result['warnings'].append(f'Dataset contains {validation_result["summary"]["duplicate_rows"]} duplicate rows')
    
    return validation_result

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     sample_data = {
#         'A': [1, 2, None, 4, 5, 5],
#         'B': [10, 20, 30, None, 50, 50],
#         'C': ['x', 'y', 'z', 'x', 'y', 'y']
#     }
#     
#     df = pd.DataFrame(sample_data)
#     
#     # Validate data
#     validation = validate_dataframe(df, required_columns=['A', 'B', 'C'])
#     print("Validation Result:", validation)
#     
#     # Clean data
#     cleaned = clean_dataset(df, remove_duplicates=True, fill_method='fill_mean')
#     print("\nCleaned DataFrame:")
#     print(cleaned)
import pandas as pd
import numpy as np
from scipy import stats

def load_data(filepath):
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_column(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(input_file, output_file):
    df = load_data(input_file)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
    
    for col in numeric_columns:
        df = normalize_column(df, col)
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    return df

if __name__ == "__main__":
    cleaned_df = clean_dataset('raw_data.csv', 'cleaned_data.csv')import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    filtered_indices = np.where(z_scores < threshold)[0]
    
    filtered_data = data.iloc[filtered_indices].reset_index(drop=True)
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
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Main function to clean dataset with outlier removal and normalization
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
            cleaned_data[f'{column}_standardized'] = normalize_zscore(cleaned_data, column)
    
    return cleaned_data

def validate_data(data, required_columns=None, check_missing=True, check_duplicates=True):
    """
    Validate dataset for common data quality issues
    """
    validation_report = {}
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        validation_report['missing_columns'] = missing_columns
    
    if check_missing:
        missing_values = data.isnull().sum()
        validation_report['missing_values'] = missing_values[missing_values > 0].to_dict()
    
    if check_duplicates:
        duplicate_count = data.duplicated().sum()
        validation_report['duplicate_rows'] = duplicate_count
    
    return validation_reportimport numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, threshold=1.5):
    """
    Detect outliers using Interquartile Range method.
    
    Parameters:
    data (array-like): Input data
    threshold (float): Multiplier for IQR (default 1.5)
    
    Returns:
    tuple: (lower_bound, upper_bound, outlier_indices)
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    outlier_mask = (data < lower_bound) | (data > upper_bound)
    outlier_indices = np.where(outlier_mask)[0]
    
    return lower_bound, upper_bound, outlier_indices

def normalize_minmax(data, feature_range=(0, 1)):
    """
    Normalize data using Min-Max scaling.
    
    Parameters:
    data (array-like): Input data
    feature_range (tuple): Desired range of transformed data
    
    Returns:
    array: Normalized data
    """
    data_min = np.min(data)
    data_max = np.max(data)
    
    if data_max == data_min:
        return np.zeros_like(data)
    
    normalized = (data - data_min) / (data_max - data_min)
    min_val, max_val = feature_range
    return normalized * (max_val - min_val) + min_val

def remove_outliers_zscore(data, threshold=3):
    """
    Remove outliers using Z-score method.
    
    Parameters:
    data (array-like): Input data
    threshold (float): Z-score threshold (default 3)
    
    Returns:
    array: Data with outliers removed
    """
    z_scores = np.abs(stats.zscore(data))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def clean_dataframe(df, numeric_columns=None, outlier_method='iqr'):
    """
    Clean dataframe by handling outliers in numeric columns.
    
    Parameters:
    df (DataFrame): Input dataframe
    numeric_columns (list): List of numeric column names
    outlier_method (str): Method for outlier detection ('iqr' or 'zscore')
    
    Returns:
    DataFrame: Cleaned dataframe
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col not in df.columns:
            continue
            
        col_data = df[col].dropna().values
        
        if outlier_method == 'iqr':
            _, _, outlier_idx = detect_outliers_iqr(col_data)
        elif outlier_method == 'zscore':
            z_scores = np.abs(stats.zscore(col_data))
            outlier_idx = np.where(z_scores > 3)[0]
        else:
            raise ValueError(f"Unknown outlier method: {outlier_method}")
        
        if len(outlier_idx) > 0:
            median_val = np.median(col_data)
            cleaned_df.loc[cleaned_df.index[outlier_idx], col] = median_val
    
    return cleaned_df

def create_cleaning_pipeline(steps):
    """
    Create a data cleaning pipeline with specified steps.
    
    Parameters:
    steps (list): List of cleaning function names
    
    Returns:
    function: Pipeline function
    """
    available_steps = {
        'remove_outliers': remove_outliers_zscore,
        'normalize': normalize_minmax,
        'detect_outliers': detect_outliers_iqr
    }
    
    def pipeline(data, **kwargs):
        result = data.copy()
        
        for step in steps:
            if step in available_steps:
                if step == 'remove_outliers':
                    result = available_steps[step](result, **kwargs.get(step, {}))
                elif step == 'normalize':
                    result = available_steps[step](result, **kwargs.get(step, {}))
                elif step == 'detect_outliers':
                    bounds = available_steps[step](result, **kwargs.get(step, {}))
                    print(f"Outlier bounds: {bounds[:2]}")
        
        return result
    
    return pipeline

if __name__ == "__main__":
    # Example usage
    sample_data = np.random.normal(0, 1, 100)
    sample_data[0] = 10  # Add an outlier
    
    print("Original data shape:", sample_data.shape)
    print("Detecting outliers...")
    lower, upper, outliers = detect_outliers_iqr(sample_data)
    print(f"Outliers found at indices: {outliers}")
    
    cleaned_data = remove_outliers_zscore(sample_data)
    print("Cleaned data shape:", cleaned_data.shape)
    
    normalized_data = normalize_minmax(cleaned_data)
    print("Normalized data range:", normalized_data.min(), normalized_data.max())