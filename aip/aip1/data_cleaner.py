
import numpy as np
import pandas as pd

def remove_outliers_iqr(dataframe, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        dataframe: pandas DataFrame
        column: column name to process
        multiplier: IQR multiplier for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    return dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]

def normalize_minmax(dataframe, columns=None):
    """
    Normalize specified columns using Min-Max scaling.
    
    Args:
        dataframe: pandas DataFrame
        columns: list of column names to normalize (default: all numeric columns)
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns
    
    normalized_df = dataframe.copy()
    for col in columns:
        if col in dataframe.columns and np.issubdtype(dataframe[col].dtype, np.number):
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            if max_val != min_val:
                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
    
    return normalized_df

def clean_dataset(dataframe, numeric_columns=None, outlier_multiplier=1.5):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        dataframe: pandas DataFrame to clean
        numeric_columns: list of numeric columns to process
        outlier_multiplier: IQR multiplier for outlier removal
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = dataframe.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_multiplier)
    
    cleaned_df = normalize_minmax(cleaned_df, numeric_columns)
    
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(dataframe, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        dataframe: pandas DataFrame to validate
        required_columns: list of required column names
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(dataframe, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if dataframe.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in dataframe.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    print("Original shape:", sample_data.shape)
    
    is_valid, message = validate_dataframe(sample_data, ['feature_a', 'feature_b'])
    print(f"Validation: {message}")
    
    if is_valid:
        cleaned_data = clean_dataset(sample_data, ['feature_a', 'feature_b'])
        print("Cleaned shape:", cleaned_data.shape)
        print("Cleaned data summary:")
        print(cleaned_data[['feature_a', 'feature_b']].describe())import pandas as pd
import numpy as np

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

def clean_dataset(input_path, output_path):
    df = pd.read_csv(input_path)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
        df = normalize_column(df, col)
    
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    print(f"Original shape: {pd.read_csv(input_path).shape}")
    print(f"Cleaned shape: {df.shape}")

if __name__ == "__main__":
    clean_dataset('raw_data.csv', 'cleaned_data.csv')
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    multiplier (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
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
    df_copy = dataframe.copy()
    
    if columns is None:
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for col in columns:
        if col not in df_copy.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        
        if not np.issubdtype(df_copy[col].dtype, np.number):
            raise ValueError(f"Column '{col}' is not numeric")
        
        col_min = df_copy[col].min()
        col_max = df_copy[col].max()
        
        if col_max == col_min:
            df_copy[col] = 0.0
        else:
            df_copy[col] = (df_copy[col] - col_min) / (col_max - col_min)
    
    return df_copy

def zscore_normalize(dataframe, columns=None):
    """
    Normalize specified columns using z-score normalization.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    columns (list): List of column names to normalize. If None, normalize all numeric columns.
    
    Returns:
    pd.DataFrame: DataFrame with z-score normalized columns
    """
    df_copy = dataframe.copy()
    
    if columns is None:
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for col in columns:
        if col not in df_copy.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        
        if not np.issubdtype(df_copy[col].dtype, np.number):
            raise ValueError(f"Column '{col}' is not numeric")
        
        mean_val = df_copy[col].mean()
        std_val = df_copy[col].std()
        
        if std_val == 0:
            df_copy[col] = 0.0
        else:
            df_copy[col] = (df_copy[col] - mean_val) / std_val
    
    return df_copy

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    columns (list): List of column names to process. If None, process all columns.
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    df_copy = dataframe.copy()
    
    if columns is None:
        columns = df_copy.columns
    
    for col in columns:
        if col not in df_copy.columns:
            continue
        
        if df_copy[col].isnull().sum() == 0:
            continue
        
        if strategy == 'drop':
            df_copy = df_copy.dropna(subset=[col])
        elif strategy == 'mean':
            if np.issubdtype(df_copy[col].dtype, np.number):
                df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
        elif strategy == 'median':
            if np.issubdtype(df_copy[col].dtype, np.number):
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())
        elif strategy == 'mode':
            mode_val = df_copy[col].mode()
            if not mode_val.empty:
                df_copy[col] = df_copy[col].fillna(mode_val.iloc[0])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return df_copy

def clean_dataset(dataframe, numeric_columns=None, outlier_multiplier=1.5, 
                  normalize_method='minmax', missing_strategy='mean'):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric columns to process
    outlier_multiplier (float): IQR multiplier for outlier detection
    normalize_method (str): Normalization method ('minmax', 'zscore', or None)
    missing_strategy (str): Strategy for handling missing values
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    df_copy = dataframe.copy()
    
    if numeric_columns is None:
        numeric_columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    
    df_copy = handle_missing_values(df_copy, strategy=missing_strategy, columns=numeric_columns)
    
    for col in numeric_columns:
        if col in df_copy.columns:
            df_copy = remove_outliers_iqr(df_copy, col, multiplier=outlier_multiplier)
    
    if normalize_method == 'minmax':
        df_copy = normalize_minmax(df_copy, columns=numeric_columns)
    elif normalize_method == 'zscore':
        df_copy = zscore_normalize(df_copy, columns=numeric_columns)
    
    return df_copy.reset_index(drop=True)import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    subset (list, optional): Column labels to consider for duplicates.
    keep (str, optional): Which duplicates to keep.
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed.
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    return cleaned_df

def clean_numeric_column(df, column_name, fill_method='mean'):
    """
    Clean a numeric column by filling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column_name (str): Name of the column to clean.
    fill_method (str): Method to fill missing values ('mean', 'median', 'mode').
    
    Returns:
    pd.DataFrame: DataFrame with cleaned column.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    if df[column_name].dtype not in ['int64', 'float64']:
        raise TypeError(f"Column '{column_name}' must be numeric")
    
    df_copy = df.copy()
    
    if fill_method == 'mean':
        fill_value = df_copy[column_name].mean()
    elif fill_method == 'median':
        fill_value = df_copy[column_name].median()
    elif fill_method == 'mode':
        fill_value = df_copy[column_name].mode()[0]
    else:
        raise ValueError("fill_method must be 'mean', 'median', or 'mode'")
    
    df_copy[column_name] = df_copy[column_name].fillna(fill_value)
    return df_copy

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
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
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"import re
import pandas as pd

def remove_duplicates(dataframe, subset=None):
    """
    Remove duplicate rows from a DataFrame.
    """
    return dataframe.drop_duplicates(subset=subset, keep='first')

def validate_email(email):
    """
    Validate an email address format.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def clean_column_names(dataframe):
    """
    Clean column names by removing whitespace and converting to lowercase.
    """
    dataframe.columns = dataframe.columns.str.strip().str.lower()
    return dataframe

def fill_missing_values(dataframe, column, method='mean'):
    """
    Fill missing values in a column with mean, median, or mode.
    """
    if method == 'mean':
        fill_value = dataframe[column].mean()
    elif method == 'median':
        fill_value = dataframe[column].median()
    elif method == 'mode':
        fill_value = dataframe[column].mode()[0]
    else:
        fill_value = 0
    dataframe[column] = dataframe[column].fillna(fill_value)
    return dataframe

def convert_to_datetime(dataframe, column, format=None):
    """
    Convert a column to datetime format.
    """
    if format:
        dataframe[column] = pd.to_datetime(dataframe[column], format=format)
    else:
        dataframe[column] = pd.to_datetime(dataframe[column])
    return dataframe

def filter_by_range(dataframe, column, min_val, max_val):
    """
    Filter rows where column values are within a specified range.
    """
    return dataframe[(dataframe[column] >= min_val) & (dataframe[column] <= max_val)]import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Method to fill missing values. Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
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
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            elif fill_missing == 'median':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            mode_val = cleaned_df[col].mode()
            if not mode_val.empty:
                cleaned_df[col] = cleaned_df[col].fillna(mode_val.iloc[0])
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for required columns and data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names. Default is None.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    return True, "Dataset is valid"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', 'z', 'z']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_dataset(cleaned, required_columns=['A', 'B'])
    print(f"\nValidation: {message}")