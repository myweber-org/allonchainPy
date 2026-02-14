
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(filepath):
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
    df = load_dataset(input_file)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
    
    for col in numeric_columns:
        df = normalize_column(df, col)
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    return df

if __name__ == "__main__":
    cleaned_df = clean_dataset('raw_data.csv', 'cleaned_data.csv')import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str): Strategy to fill missing values ('mean', 'median', 'mode', or 'drop').
    
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
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
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
    
    return True, "Data validation passed"

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
    
    is_valid, message = validate_data(cleaned, required_columns=['A', 'B'])
    print(f"\nValidation: {is_valid}, Message: {message}")
import pandas as pd

def clean_dataset(df, drop_na=True, rename_columns=True):
    """
    Clean a pandas DataFrame by handling missing values and standardizing column names.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_na (bool): If True, drop rows with any null values.
    rename_columns (bool): If True, rename columns to lowercase with underscores.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_na:
        cleaned_df = cleaned_df.dropna()
    
    if rename_columns:
        cleaned_df.columns = (
            cleaned_df.columns
            .str.lower()
            .str.replace(' ', '_')
            .str.replace(r'[^\w_]', '', regex=True)
        )
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate the DataFrame for required columns and data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    if df.empty:
        print("DataFrame is empty")
        return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'Product Name': ['Widget A', 'Widget B', None, 'Widget C'],
        'Price ($)': [10.99, 15.49, 12.99, None],
        'Quantity': [100, 150, 200, 250]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned = clean_dataset(df)
    print(cleaned)
    print(f"\nData validation: {validate_data(cleaned)}")
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Args:
        dataframe: pandas DataFrame
        column: column name to process
        threshold: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
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
    
    return filtered_df

def zscore_normalize(dataframe, columns=None):
    """
    Normalize specified columns using z-score normalization.
    
    Args:
        dataframe: pandas DataFrame
        columns: list of column names to normalize (default: all numeric columns)
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col in normalized_df.columns and pd.api.types.is_numeric_dtype(normalized_df[col]):
            mean_val = normalized_df[col].mean()
            std_val = normalized_df[col].std()
            
            if std_val > 0:
                normalized_df[col] = (normalized_df[col] - mean_val) / std_val
            else:
                normalized_df[col] = 0
    
    return normalized_df

def minmax_normalize(dataframe, columns=None, feature_range=(0, 1)):
    """
    Normalize specified columns using min-max normalization.
    
    Args:
        dataframe: pandas DataFrame
        columns: list of column names to normalize
        feature_range: tuple of (min, max) for output range
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    normalized_df = dataframe.copy()
    min_val, max_val = feature_range
    
    for col in columns:
        if col in normalized_df.columns and pd.api.types.is_numeric_dtype(normalized_df[col]):
            col_min = normalized_df[col].min()
            col_max = normalized_df[col].max()
            col_range = col_max - col_min
            
            if col_range > 0:
                normalized_df[col] = ((normalized_df[col] - col_min) / col_range) * (max_val - min_val) + min_val
            else:
                normalized_df[col] = min_val
    
    return normalized_df

def detect_skewed_columns(dataframe, threshold=0.5):
    """
    Detect columns with significant skewness.
    
    Args:
        dataframe: pandas DataFrame
        threshold: absolute skewness threshold (default 0.5)
    
    Returns:
        Dictionary of column names and their skewness values
    """
    skewed_cols = {}
    
    for col in dataframe.select_dtypes(include=[np.number]).columns:
        skewness = dataframe[col].skew()
        if abs(skewness) > threshold:
            skewed_cols[col] = skewness
    
    return skewed_cols

def log_transform(dataframe, columns):
    """
    Apply log transformation to specified columns.
    
    Args:
        dataframe: pandas DataFrame
        columns: list of column names to transform
    
    Returns:
        DataFrame with transformed columns
    """
    transformed_df = dataframe.copy()
    
    for col in columns:
        if col in transformed_df.columns and pd.api.types.is_numeric_dtype(transformed_df[col]):
            # Add small constant to handle zero values
            min_val = transformed_df[col].min()
            if min_val <= 0:
                constant = abs(min_val) + 1
                transformed_df[col] = np.log(transformed_df[col] + constant)
            else:
                transformed_df[col] = np.log(transformed_df[col])
    
    return transformed_df

def clean_dataset(dataframe, outlier_columns=None, normalize_method='zscore', 
                  normalize_columns=None, handle_skewness=True):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        dataframe: pandas DataFrame
        outlier_columns: columns to remove outliers from
        normalize_method: 'zscore', 'minmax', or None
        normalize_columns: columns to normalize
        handle_skewness: whether to apply log transform to skewed columns
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = dataframe.copy()
    
    # Remove outliers
    if outlier_columns:
        for col in outlier_columns:
            if col in cleaned_df.columns:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    # Handle skewness
    if handle_skewness:
        skewed_cols = detect_skewed_columns(cleaned_df)
        if skewed_cols:
            skewed_cols_list = list(skewed_cols.keys())
            cleaned_df = log_transform(cleaned_df, skewed_cols_list)
    
    # Normalize
    if normalize_method == 'zscore':
        cleaned_df = zscore_normalize(cleaned_df, normalize_columns)
    elif normalize_method == 'minmax':
        cleaned_df = minmax_normalize(cleaned_df, normalize_columns)
    
    return cleaned_df

def validate_dataframe(dataframe, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        dataframe: pandas DataFrame to validate
        required_columns: list of required column names
        min_rows: minimum number of rows required
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(dataframe, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(dataframe) < min_rows:
        return False, f"DataFrame must have at least {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in dataframe.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"