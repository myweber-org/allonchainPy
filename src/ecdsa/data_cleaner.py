
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    threshold (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df.copy()

def normalize_minmax(dataframe, columns=None):
    """
    Normalize specified columns using min-max scaling.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    columns (list): List of column names to normalize. If None, normalize all numeric columns.
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    if columns is None:
        numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col not in normalized_df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        
        if not pd.api.types.is_numeric_dtype(normalized_df[col]):
            raise ValueError(f"Column '{col}' is not numeric")
        
        col_min = normalized_df[col].min()
        col_max = normalized_df[col].max()
        
        if col_max == col_min:
            normalized_df[col] = 0.5
        else:
            normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
    
    return normalized_df

def zscore_normalize(dataframe, columns=None):
    """
    Normalize specified columns using z-score normalization.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    columns (list): List of column names to normalize. If None, normalize all numeric columns.
    
    Returns:
    pd.DataFrame: DataFrame with z-score normalized columns
    """
    if columns is None:
        numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col not in normalized_df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        
        if not pd.api.types.is_numeric_dtype(normalized_df[col]):
            raise ValueError(f"Column '{col}' is not numeric")
        
        mean_val = normalized_df[col].mean()
        std_val = normalized_df[col].std()
        
        if std_val == 0:
            normalized_df[col] = 0
        else:
            normalized_df[col] = (normalized_df[col] - mean_val) / std_val
    
    return normalized_df

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    strategy (str): Strategy for imputation ('mean', 'median', 'mode', 'drop')
    columns (list): List of column names to process. If None, process all columns.
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    if columns is None:
        columns = dataframe.columns
    
    processed_df = dataframe.copy()
    
    for col in columns:
        if col not in processed_df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        
        if processed_df[col].isnull().any():
            if strategy == 'drop':
                processed_df = processed_df.dropna(subset=[col])
            elif strategy == 'mean':
                if pd.api.types.is_numeric_dtype(processed_df[col]):
                    processed_df[col].fillna(processed_df[col].mean(), inplace=True)
            elif strategy == 'median':
                if pd.api.types.is_numeric_dtype(processed_df[col]):
                    processed_df[col].fillna(processed_df[col].median(), inplace=True)
            elif strategy == 'mode':
                processed_df[col].fillna(processed_df[col].mode()[0], inplace=True)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
    
    return processed_df

def clean_dataset(dataframe, outlier_columns=None, normalize_method='minmax', missing_strategy='mean'):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    outlier_columns (list): Columns to apply outlier removal
    normalize_method (str): Normalization method ('minmax', 'zscore', or None)
    missing_strategy (str): Strategy for handling missing values
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = dataframe.copy()
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    if outlier_columns:
        for col in outlier_columns:
            if col in cleaned_df.columns:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    if normalize_method == 'minmax':
        cleaned_df = normalize_minmax(cleaned_df)
    elif normalize_method == 'zscore':
        cleaned_df = zscore_normalize(cleaned_df)
    
    return cleaned_df.reset_index(drop=True)
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_method=None):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to drop duplicate rows
        fill_method (str or None): Method to fill missing values: 
                                  'mean', 'median', 'mode', or None to drop
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_method:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        
        if fill_method == 'mean':
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(
                cleaned_df[numeric_cols].mean()
            )
        elif fill_method == 'median':
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(
                cleaned_df[numeric_cols].median()
            )
        elif fill_method == 'mode':
            for col in cleaned_df.columns:
                mode_val = cleaned_df[col].mode()
                if not mode_val.empty:
                    cleaned_df[col] = cleaned_df[col].fillna(mode_val[0])
    else:
        # Drop rows with any missing values
        cleaned_df = cleaned_df.dropna()
    
    # Remove duplicates
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate dataset structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'row_count': len(df),
        'column_count': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        validation_results['missing_columns'] = missing_cols
        validation_results['all_columns_present'] = len(missing_cols) == 0
    
    return validation_results
import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def handle_missing_values(df, strategy='drop', fill_value=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        df: pandas DataFrame
        strategy: 'drop' to remove rows, 'fill' to replace values
        fill_value: value to use when strategy is 'fill'
    
    Returns:
        DataFrame with handled missing values
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill':
        if fill_value is not None:
            return df.fillna(fill_value)
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            return df
    else:
        raise ValueError("Strategy must be 'drop' or 'fill'")

def clean_column_names(df):
    """
    Standardize column names to lowercase with underscores.
    
    Args:
        df: pandas DataFrame
    
    Returns:
        DataFrame with cleaned column names
    """
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df

def validate_data(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame
        required_columns: list of column names that must be present
    
    Returns:
        tuple of (is_valid, error_message)
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    return True, "Data validation passed"

def process_dataframe(df, cleaning_steps=None):
    """
    Apply multiple cleaning steps to DataFrame.
    
    Args:
        df: pandas DataFrame
        cleaning_steps: list of cleaning function names to apply
    
    Returns:
        Cleaned DataFrame
    """
    if cleaning_steps is None:
        cleaning_steps = ['clean_column_names', 'remove_duplicates', 'handle_missing_values']
    
    result_df = df.copy()
    
    for step in cleaning_steps:
        if step == 'clean_column_names':
            result_df = clean_column_names(result_df)
        elif step == 'remove_duplicates':
            result_df = remove_duplicates(result_df)
        elif step == 'handle_missing_values':
            result_df = handle_missing_values(result_df, strategy='fill')
    
    return result_dfimport pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): If True, remove duplicate rows.
    fill_missing (str): Method to fill missing values. Options: 'mean', 'median', 'mode', or 'drop'.
    
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
    Validate a DataFrame for basic integrity checks.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (bool, str) indicating validation result and message.
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 30, 40, 50],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned)
    print(f"\nValidation: {is_valid} - {message}")
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (list, optional): Column labels to consider for duplicates
        keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by converting to appropriate types.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to clean
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns
    """
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of required column names
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
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

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns=None, outlier_multiplier=1.5):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process (default: all numeric)
        outlier_multiplier: multiplier for IQR outlier detection
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_data = data.copy()
    
    if numeric_columns is None:
        numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
    
    for column in numeric_columns:
        if column in cleaned_data.columns:
            # Remove outliers
            q1 = cleaned_data[column].quantile(0.25)
            q3 = cleaned_data[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - outlier_multiplier * iqr
            upper_bound = q3 + outlier_multiplier * iqr
            
            mask = (cleaned_data[column] >= lower_bound) & (cleaned_data[column] <= upper_bound)
            cleaned_data = cleaned_data[mask]
            
            # Normalize the column
            min_val = cleaned_data[column].min()
            max_val = cleaned_data[column].max()
            if max_val > min_val:
                cleaned_data[f"{column}_normalized"] = (cleaned_data[column] - min_val) / (max_val - min_val)
    
    return cleaned_data

def validate_data(data, required_columns=None, allow_nan=True):
    """
    Validate dataset structure and content.
    
    Args:
        data: pandas DataFrame to validate
        required_columns: list of required column names
        allow_nan: whether NaN values are allowed
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(data, pd.DataFrame):
        return False, "Input must be a pandas DataFrame"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if not allow_nan and data.isnull().any().any():
        return False, "Dataset contains NaN values"
    
    return True, "Dataset is valid"
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

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a DataFrame column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to analyze
    
    Returns:
        dict: Dictionary containing statistical measures
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def process_dataframe(df, column):
    """
    Process DataFrame by removing outliers and returning statistics.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to process
    
    Returns:
        tuple: (cleaned_df, original_stats, cleaned_stats)
    """
    original_stats = calculate_statistics(df, column)
    cleaned_df = remove_outliers_iqr(df, column)
    cleaned_stats = calculate_statistics(cleaned_df, column)
    
    return cleaned_df, original_stats, cleaned_stats

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 102, 12, 14, 13, 12, 11, 14, 13, 12, 14, 100]
    }
    
    df = pd.DataFrame(sample_data)
    
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned_df, original_stats, cleaned_stats = process_dataframe(df, 'values')
    
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print()
    
    print("Original Statistics:")
    for key, value in original_stats.items():
        print(f"{key}: {value}")
    
    print()
    print("Cleaned Statistics:")
    for key, value in cleaned_stats.items():
        print(f"{key}: {value}")
import pandas as pd
import numpy as np

def clean_dataset(df, text_columns=None, fill_strategy='mean'):
    """
    Clean a pandas DataFrame by handling missing values and standardizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        text_columns (list): List of column names containing text data
        fill_strategy (str): Strategy for filling numeric missing values ('mean', 'median', 'mode')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Identify numeric and text columns
    if text_columns is None:
        text_columns = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    # Handle missing values in numeric columns
    for col in numeric_columns:
        if df_clean[col].isnull().any():
            if fill_strategy == 'mean':
                fill_value = df_clean[col].mean()
            elif fill_strategy == 'median':
                fill_value = df_clean[col].median()
            elif fill_strategy == 'mode':
                fill_value = df_clean[col].mode()[0]
            else:
                fill_value = 0
            
            df_clean[col].fillna(fill_value, inplace=True)
    
    # Standardize text columns
    for col in text_columns:
        if col in df_clean.columns:
            # Convert to string, strip whitespace, and standardize case
            df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
            # Replace empty strings with NaN
            df_clean[col] = df_clean[col].replace(['', 'nan', 'none'], np.nan)
    
    # Remove duplicate rows
    df_clean = df_clean.drop_duplicates()
    
    # Reset index after cleaning
    df_clean.reset_index(drop=True, inplace=True)
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of columns that must be present
    
    Returns:
        dict: Validation results with status and messages
    """
    validation_result = {
        'is_valid': True,
        'messages': [],
        'missing_columns': [],
        'null_counts': {}
    }
    
    if not isinstance(df, pd.DataFrame):
        validation_result['is_valid'] = False
        validation_result['messages'].append('Input is not a pandas DataFrame')
        return validation_result
    
    # Check required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_result['is_valid'] = False
            validation_result['missing_columns'] = missing_cols
            validation_result['messages'].append(f'Missing required columns: {missing_cols}')
    
    # Check for null values
    null_counts = df.isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0]
    
    if not columns_with_nulls.empty:
        validation_result['null_counts'] = columns_with_nulls.to_dict()
        validation_result['messages'].append(f'Found null values in {len(columns_with_nulls)} columns')
    
    return validation_result

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = {
#         'name': ['Alice', 'Bob', 'Charlie', None, 'Eve'],
#         'age': [25, 30, None, 35, 40],
#         'score': [85.5, 92.0, 78.5, None, 88.0],
#         'city': ['New York', 'los angeles', '  Chicago  ', '', 'Boston']
#     }
#     
#     df = pd.DataFrame(sample_data)
#     print("Original DataFrame:")
#     print(df)
#     print("\n" + "="*50 + "\n")
#     
#     cleaned_df = clean_dataset(df, text_columns=['name', 'city'])
#     print("Cleaned DataFrame:")
#     print(cleaned_df)
#     
#     validation = validate_dataframe(cleaned_df, required_columns=['name', 'age', 'score'])
#     print("\nValidation Results:")
#     print(validation)