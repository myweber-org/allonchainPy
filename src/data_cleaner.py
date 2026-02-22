
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None, remove_duplicates=True, case_normalization='lower'):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    columns_to_clean (list, optional): List of column names to apply string normalization.
                                       If None, applies to all object dtype columns.
    remove_duplicates (bool): If True, remove duplicate rows.
    case_normalization (str): One of 'lower', 'upper', or None for case normalization.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if remove_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if columns_to_clean is None:
        columns_to_clean = cleaned_df.select_dtypes(include=['object']).columns.tolist()
    
    for col in columns_to_clean:
        if col in cleaned_df.columns and cleaned_df[col].dtype == 'object':
            cleaned_df[col] = cleaned_df[col].astype(str)
            
            if case_normalization == 'lower':
                cleaned_df[col] = cleaned_df[col].str.lower()
            elif case_normalization == 'upper':
                cleaned_df[col] = cleaned_df[col].str.upper()
            
            cleaned_df[col] = cleaned_df[col].apply(lambda x: re.sub(r'\s+', ' ', x.strip()))
            
            print(f"Cleaned column: {col}")
    
    return cleaned_df

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    email_column (str): Name of the column containing email addresses.
    
    Returns:
    pd.DataFrame: DataFrame with additional 'email_valid' boolean column.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame.")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['email_valid'] = df[email_column].str.match(email_pattern, na=False)
    
    valid_count = df['email_valid'].sum()
    print(f"Found {valid_count} valid email addresses out of {len(df)} rows.")
    
    return df

def save_cleaned_data(df, output_path, format='csv'):
    """
    Save cleaned DataFrame to file.
    
    Parameters:
    df (pd.DataFrame): DataFrame to save.
    output_path (str): Path to save the file.
    format (str): File format ('csv', 'excel', or 'json').
    """
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'excel':
        df.to_excel(output_path, index=False)
    elif format == 'json':
        df.to_json(output_path, orient='records')
    else:
        raise ValueError("Unsupported format. Use 'csv', 'excel', or 'json'.")
    
    print(f"Data saved to {output_path} in {format} format.")
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
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

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
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

def normalize_column(df, column, method='minmax'):
    """
    Normalize a DataFrame column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        if max_val != min_val:
            df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val != 0:
            df_copy[column] = (df_copy[column] - mean_val) / std_val
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df_copyimport pandas as pd
import re

def clean_text_column(df, column_name):
    """
    Standardize text by converting to lowercase, removing extra whitespace,
    and stripping special characters (except basic punctuation).
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = df[column_name].astype(str)
    df[column_name] = df[column_name].str.lower()
    df[column_name] = df[column_name].str.strip()
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'[^\w\s.,!?-]', '', x))
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'\s+', ' ', x))
    
    return df

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def validate_email_column(df, column_name):
    """
    Validate email format in a specified column and return a boolean series.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return df[column_name].astype(str).str.match(email_pattern)

def sample_data_cleaning():
    """
    Example usage of the cleaning functions.
    """
    data = {
        'name': ['Alice', 'Bob', 'Alice', 'Charlie  ', 'david'],
        'email': ['alice@example.com', 'invalid-email', 'alice@example.com', 'charlie@test.org', 'david@domain.co']
    }
    df = pd.DataFrame(data)
    
    print("Original DataFrame:")
    print(df)
    
    df = clean_text_column(df, 'name')
    df = remove_duplicates(df, subset=['email'], keep='first')
    df['valid_email'] = validate_email_column(df, 'email')
    
    print("\nCleaned DataFrame:")
    print(df)
    
    return df

if __name__ == "__main__":
    sample_data_cleaning()
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
        columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col in dataframe.columns and np.issubdtype(dataframe[col].dtype, np.number):
            col_min = dataframe[col].min()
            col_max = dataframe[col].max()
            
            if col_max != col_min:
                normalized_df[col] = (dataframe[col] - col_min) / (col_max - col_min)
            else:
                normalized_df[col] = 0
    
    return normalized_df

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame columns.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    strategy (str): Imputation strategy ('mean', 'median', 'mode', or 'drop')
    columns (list): List of column names to process. If None, process all columns.
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    if columns is None:
        columns = dataframe.columns.tolist()
    
    processed_df = dataframe.copy()
    
    for col in columns:
        if col not in processed_df.columns:
            continue
            
        missing_mask = processed_df[col].isna()
        
        if not missing_mask.any():
            continue
            
        if strategy == 'drop':
            processed_df = processed_df.dropna(subset=[col])
        elif strategy == 'mean':
            if np.issubdtype(processed_df[col].dtype, np.number):
                fill_value = processed_df[col].mean()
                processed_df[col] = processed_df[col].fillna(fill_value)
        elif strategy == 'median':
            if np.issubdtype(processed_df[col].dtype, np.number):
                fill_value = processed_df[col].median()
                processed_df[col] = processed_df[col].fillna(fill_value)
        elif strategy == 'mode':
            fill_value = processed_df[col].mode()[0] if not processed_df[col].mode().empty else None
            processed_df[col] = processed_df[col].fillna(fill_value)
    
    return processed_df

def calculate_descriptive_stats(dataframe, columns=None):
    """
    Calculate descriptive statistics for specified columns.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    columns (list): List of column names to analyze. If None, analyze all numeric columns.
    
    Returns:
    pd.DataFrame: DataFrame with descriptive statistics
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    stats_dict = {}
    
    for col in columns:
        if col in dataframe.columns and np.issubdtype(dataframe[col].dtype, np.number):
            col_data = dataframe[col].dropna()
            
            if len(col_data) > 0:
                stats_dict[col] = {
                    'mean': np.mean(col_data),
                    'median': np.median(col_data),
                    'std': np.std(col_data),
                    'min': np.min(col_data),
                    'max': np.max(col_data),
                    'q1': np.percentile(col_data, 25),
                    'q3': np.percentile(col_data, 75),
                    'skewness': stats.skew(col_data),
                    'kurtosis': stats.kurtosis(col_data)
                }
    
    stats_df = pd.DataFrame(stats_dict).T
    return stats_df

def validate_dataframe(dataframe, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    dataframe (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, message)
    """
    if not isinstance(dataframe, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(dataframe) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for identifying duplicates
        keep: determines which duplicates to keep ('first', 'last', or False)
    
    Returns:
        DataFrame with duplicates removed
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
    Clean numeric columns by converting to appropriate types and handling errors.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to clean
    
    Returns:
        DataFrame with cleaned numeric columns
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of columns that must be present
    
    Returns:
        Tuple of (is_valid, message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"import pandas as pd

def clean_dataset(df, sort_column=None):
    """
    Remove duplicate rows from a DataFrame and optionally sort by a column.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    sort_column (str, optional): Column name to sort by. Defaults to None.

    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.drop_duplicates().reset_index(drop=True)
    
    if sort_column and sort_column in cleaned_df.columns:
        cleaned_df = cleaned_df.sort_values(by=sort_column).reset_index(drop=True)
    
    return cleaned_df

def main():
    sample_data = {
        'id': [1, 2, 2, 3, 1],
        'value': [10, 20, 20, 30, 10],
        'name': ['A', 'B', 'B', 'C', 'A']
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, sort_column='id')
    print("\nCleaned DataFrame:")
    print(cleaned)

if __name__ == "__main__":
    main()