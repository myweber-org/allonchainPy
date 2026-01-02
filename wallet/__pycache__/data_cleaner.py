
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed from the specified column.
    """
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
    data (pd.DataFrame): The DataFrame containing the data.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing count, mean, std, min, and max.
    """
    stats = {
        'count': data[column].count(),
        'mean': data[column].mean(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max()
    }
    return stats
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        factor: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        threshold: Z-score threshold (default 3)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    mask = z_scores < threshold
    return data[mask]

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
        return data[column].apply(lambda x: 0.5)
    
    return (data[column] - min_val) / (max_val - min_val)

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    return (data[column] - mean_val) / std_val

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process (default: all numeric)
        outlier_method: 'iqr', 'zscore', or None
        normalize_method: 'minmax', 'zscore', or None
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    # Remove outliers
    if outlier_method == 'iqr':
        for col in numeric_columns:
            if col in cleaned_data.columns:
                cleaned_data = remove_outliers_iqr(cleaned_data, col)
    elif outlier_method == 'zscore':
        for col in numeric_columns:
            if col in cleaned_data.columns:
                cleaned_data = remove_outliers_zscore(cleaned_data, col)
    
    # Normalize data
    if normalize_method == 'minmax':
        for col in numeric_columns:
            if col in cleaned_data.columns:
                cleaned_data[col] = normalize_minmax(cleaned_data, col)
    elif normalize_method == 'zscore':
        for col in numeric_columns:
            if col in cleaned_data.columns:
                cleaned_data[col] = normalize_zscore(cleaned_data, col)
    
    return cleaned_data

def validate_data(data, required_columns=None, allow_nan=False):
    """
    Validate dataset structure and content.
    
    Args:
        data: pandas DataFrame
        required_columns: list of required column names
        allow_nan: whether to allow NaN values
    
    Returns:
        tuple: (is_valid, message)
    """
    if not isinstance(data, pd.DataFrame):
        return False, "Input must be a pandas DataFrame"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    if not allow_nan and data.isnull().any().any():
        nan_cols = data.columns[data.isnull().any()].tolist()
        return False, f"NaN values found in columns: {nan_cols}"
    
    return True, "Data validation passed"
import pandas as pd
import numpy as np

def clean_dataset(df, column_mapping=None, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by handling duplicates, missing values, and column renaming.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        column_mapping (dict, optional): Dictionary mapping old column names to new ones.
        drop_duplicates (bool): Whether to drop duplicate rows.
        fill_missing (str): Strategy to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    if column_mapping:
        df_clean = df_clean.rename(columns=column_mapping)
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()
    
    if fill_missing == 'drop':
        df_clean = df_clean.dropna()
    else:
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            if df_clean[col].isnull().any():
                if fill_missing == 'mean':
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                elif fill_missing == 'median':
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                elif fill_missing == 'mode':
                    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    
    for col in df_clean.select_dtypes(include=['object']).columns:
        if df_clean[col].isnull().any():
            df_clean[col].fillna('Unknown', inplace=True)
    
    return df_clean

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
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
    
    return True, "Data validation passed"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 1],
        'B': [5, 6, 7, np.nan, 5],
        'C': ['x', 'y', 'z', None, 'x']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned_df = clean_dataset(df, column_mapping={'A': 'Column_A', 'B': 'Column_B'}, 
                               drop_duplicates=True, fill_missing='mean')
    
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n")
    
    is_valid, message = validate_data(cleaned_df, required_columns=['Column_A', 'Column_B', 'C'])
    print(f"Validation result: {is_valid}")
    print(f"Validation message: {message}")
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers from a column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to process
    threshold (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def normalize_minmax(data, columns=None):
    """
    Normalize specified columns using min-max scaling.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    columns (list): List of column names to normalize. If None, normalize all numeric columns.
    
    Returns:
    pd.DataFrame: Dataframe with normalized columns
    """
    if columns is None:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    normalized_data = data.copy()
    
    for col in columns:
        if col not in normalized_data.columns:
            continue
            
        if normalized_data[col].dtype in [np.float64, np.int64]:
            col_min = normalized_data[col].min()
            col_max = normalized_data[col].max()
            
            if col_max != col_min:
                normalized_data[col] = (normalized_data[col] - col_min) / (col_max - col_min)
            else:
                normalized_data[col] = 0
    
    return normalized_data

def clean_dataset(data, outlier_columns=None, normalize_columns=None, outlier_threshold=1.5):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    outlier_columns (list): Columns to remove outliers from
    normalize_columns (list): Columns to normalize
    outlier_threshold (float): IQR threshold for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    cleaned_data = data.copy()
    
    if outlier_columns:
        for col in outlier_columns:
            if col in cleaned_data.columns:
                cleaned_data = remove_outliers_iqr(cleaned_data, col, outlier_threshold)
    
    if normalize_columns:
        cleaned_data = normalize_minmax(cleaned_data, normalize_columns)
    
    cleaned_data = cleaned_data.reset_index(drop=True)
    return cleaned_data

def validate_data(data, required_columns=None, allow_nan=False):
    """
    Validate dataframe structure and content.
    
    Parameters:
    data (pd.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    allow_nan (bool): Whether NaN values are allowed
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(data, pd.DataFrame):
        return False, "Input must be a pandas DataFrame"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    if not allow_nan and data.isnull().any().any():
        return False, "Dataframe contains NaN values"
    
    return True, "Data validation passed"