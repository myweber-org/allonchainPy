import pandas as pd
import numpy as np

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
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path):
    df = pd.read_csv(file_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        df = remove_outliers_iqr(df, col)
        df = normalize_minmax(df, col)
    
    return df

if __name__ == "__main__":
    cleaned_df = clean_dataset('sample_data.csv')
    cleaned_df.to_csv('cleaned_data.csv', index=False)
    print("Data cleaning completed. Cleaned data saved to 'cleaned_data.csv'")
import pandas as pd

def remove_duplicates(dataframe, subset=None, keep='first'):
    """
    Remove duplicate rows from a pandas DataFrame.
    
    Args:
        dataframe (pd.DataFrame): Input DataFrame
        subset (list, optional): Column labels to consider for duplicates
        keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    cleaned_df = dataframe.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(dataframe) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_numeric_columns(dataframe, columns=None):
    """
    Clean numeric columns by converting to appropriate types and handling errors.
    
    Args:
        dataframe (pd.DataFrame): Input DataFrame
        columns (list, optional): Specific columns to clean
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=['object']).columns
    
    df_copy = dataframe.copy()
    
    for col in columns:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    
    return df_copy

def validate_dataframe(dataframe, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        dataframe (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): Columns that must be present
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(dataframe, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(dataframe) == 0:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to process
    multiplier (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling to range [0, 1].
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to normalize
    
    Returns:
    pd.Series: Normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to standardize
    
    Returns:
    pd.Series: Standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    strategy (str): Imputation strategy ('mean', 'median', 'mode', 'drop')
    columns (list): List of columns to process, None for all columns
    
    Returns:
    pd.DataFrame: Dataframe with handled missing values
    """
    if columns is None:
        columns = data.columns
    
    result = data.copy()
    
    for col in columns:
        if col not in result.columns:
            continue
            
        if result[col].isnull().any():
            if strategy == 'mean':
                fill_value = result[col].mean()
            elif strategy == 'median':
                fill_value = result[col].median()
            elif strategy == 'mode':
                fill_value = result[col].mode()[0] if not result[col].mode().empty else np.nan
            elif strategy == 'drop':
                result = result.dropna(subset=[col])
                continue
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            result[col] = result[col].fillna(fill_value)
    
    return result

def validate_dataframe(data, required_columns=None, min_rows=1):
    """
    Validate dataframe structure and content.
    
    Parameters:
    data (pd.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(data, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(data) < min_rows:
        return False, f"Dataframe has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Dataframe is valid"
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
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    filtered_indices = np.where(z_scores < threshold)[0]
    
    filtered_data = data.iloc[filtered_indices].reset_index(drop=True)
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

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
    Main function to clean dataset by removing outliers and normalizing numeric columns
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    report = {
        'original_rows': len(data),
        'outliers_removed': 0,
        'normalized_columns': []
    }
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_data, removed = remove_outliers_iqr(cleaned_data, column)
        elif outlier_method == 'zscore':
            cleaned_data, removed = remove_outliers_zscore(cleaned_data, column)
        else:
            removed = 0
            
        report['outliers_removed'] += removed
        
        if normalize_method == 'minmax':
            cleaned_data[f'{column}_normalized'] = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data[f'{column}_normalized'] = normalize_zscore(cleaned_data, column)
        
        report['normalized_columns'].append(column)
    
    report['final_rows'] = len(cleaned_data)
    report['rows_removed'] = report['original_rows'] - report['final_rows']
    
    return cleaned_data, report

def validate_data(data, required_columns=None, allow_nan=False, min_rows=1):
    """
    Validate dataset structure and content
    """
    if len(data) < min_rows:
        raise ValueError(f"Dataset must have at least {min_rows} rows")
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if not allow_nan:
        nan_columns = data.columns[data.isna().any()].tolist()
        if nan_columns:
            raise ValueError(f"NaN values found in columns: {nan_columns}")
    
    return True
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to clean.
    
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
    df.loc[10, 'value'] = 500
    df.loc[20, 'value'] = -100
    
    print("Original DataFrame shape:", df.shape)
    print("Original statistics:", calculate_summary_statistics(df, 'value'))
    
    cleaned_df = remove_outliers_iqr(df, 'value')
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned statistics:", calculate_summary_statistics(cleaned_df, 'value'))
    
    return cleaned_df

if __name__ == "__main__":
    result_df = example_usage()
    print("\nFirst 5 rows of cleaned data:")
    print(result_df.head())