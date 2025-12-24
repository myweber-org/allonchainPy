
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
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Args:
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
        'count': len(df[column])
    }
    
    return stats

def process_dataset(file_path, column_to_clean):
    """
    Main function to load, clean, and analyze a dataset.
    
    Args:
        file_path (str): Path to CSV file
        column_to_clean (str): Column name to clean
    
    Returns:
        tuple: Cleaned DataFrame and statistics dictionary
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Original dataset shape: {df.shape}")
        
        cleaned_df = remove_outliers_iqr(df, column_to_clean)
        print(f"Cleaned dataset shape: {cleaned_df.shape}")
        print(f"Removed {len(df) - len(cleaned_df)} outliers")
        
        stats = calculate_summary_statistics(cleaned_df, column_to_clean)
        
        return cleaned_df, stats
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return None, None
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
        return None, None
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using Interquartile Range method.
    
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
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def zscore_normalize(data, column):
    """
    Normalize data using Z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    data_copy = data.copy()
    mean_val = data_copy[column].mean()
    std_val = data_copy[column].std()
    
    if std_val == 0:
        data_copy[f'{column}_normalized'] = 0
    else:
        data_copy[f'{column}_normalized'] = (data_copy[column] - mean_val) / std_val
    
    return data_copy

def minmax_normalize(data, column, feature_range=(0, 1)):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
        feature_range: tuple of (min, max) for output range
    
    Returns:
        DataFrame with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    data_copy = data.copy()
    min_val = data_copy[column].min()
    max_val = data_copy[column].max()
    
    if max_val == min_val:
        data_copy[f'{column}_normalized'] = feature_range[0]
    else:
        scaled = (data_copy[column] - min_val) / (max_val - min_val)
        data_copy[f'{column}_normalized'] = scaled * (feature_range[1] - feature_range[0]) + feature_range[0]
    
    return data_copy

def detect_skewed_columns(data, threshold=0.5):
    """
    Detect columns with skewed distributions.
    
    Args:
        data: pandas DataFrame
        threshold: absolute skewness threshold (default 0.5)
    
    Returns:
        List of column names with skewness above threshold
    """
    skewed_cols = []
    
    for column in data.select_dtypes(include=[np.number]).columns:
        skewness = data[column].skew()
        if abs(skewness) > threshold:
            skewed_cols.append((column, skewness))
    
    return sorted(skewed_cols, key=lambda x: abs(x[1]), reverse=True)

def log_transform(data, column):
    """
    Apply log transformation to reduce skewness.
    
    Args:
        data: pandas DataFrame
        column: column name to transform
    
    Returns:
        DataFrame with transformed column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    data_copy = data.copy()
    
    if (data_copy[column] <= 0).any():
        min_val = data_copy[column].min()
        if min_val <= 0:
            shift = abs(min_val) + 1
            data_copy[f'{column}_log'] = np.log(data_copy[column] + shift)
        else:
            data_copy[f'{column}_log'] = np.log(data_copy[column])
    else:
        data_copy[f'{column}_log'] = np.log(data_copy[column])
    
    return data_copy
import pandas as pd
import numpy as np
import sys

def clean_csv(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
        
        print(f"Original shape: {df.shape}")
        print(f"Missing values per column:")
        print(df.isnull().sum())
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
        print(f"Final shape: {df.shape}")
        
        return True
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_cleaner.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    success = clean_csv(input_file, output_file)
    sys.exit(0 if success else 1)
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to remove duplicate rows.
        fill_missing (str or dict): Method to fill missing values.
            Options: 'mean', 'median', 'mode', or a dictionary of column:value pairs.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        elif fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate that a DataFrame meets basic requirements.
    
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
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "Data validation passed"
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range (IQR) method.
    
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
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
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
        'count': len(df[column])
    }
    
    return stats

def process_dataframe(df, numeric_columns):
    """
    Process multiple numeric columns to remove outliers and return cleaned DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of column names to process
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary of statistics for each processed column
    """
    cleaned_df = df.copy()
    all_stats = {}
    
    for col in numeric_columns:
        if col in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[col]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_count - len(cleaned_df)
            
            stats = calculate_summary_statistics(cleaned_df, col)
            stats['outliers_removed'] = removed_count
            all_stats[col] = stats
    
    return cleaned_df, all_statsimport numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
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
    mask = z_scores < threshold
    
    filtered_data = data[mask]
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

def clean_dataset(df, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Main function to clean dataset by removing outliers and normalizing numeric columns
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    stats_report = {}
    
    for col in numeric_columns:
        if col not in df.columns:
            continue
            
        original_count = len(cleaned_df)
        
        if outlier_method == 'iqr':
            cleaned_df, removed = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df, removed = remove_outliers_zscore(cleaned_df, col)
        else:
            removed = 0
        
        if normalize_method == 'minmax':
            cleaned_df[f'{col}_normalized'] = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df[f'{col}_normalized'] = normalize_zscore(cleaned_df, col)
        
        stats_report[col] = {
            'original_rows': original_count,
            'rows_after_cleaning': len(cleaned_df),
            'outliers_removed': removed,
            'normalization_method': normalize_method
        }
    
    return cleaned_df, stats_report

def validate_data(df, required_columns=None, allow_nan_ratio=0.1):
    """
    Validate dataset for required columns and NaN ratio
    """
    validation_result = {
        'is_valid': True,
        'missing_columns': [],
        'high_nan_columns': [],
        'validation_errors': []
    }
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_result['missing_columns'] = missing_cols
            validation_result['is_valid'] = False
            validation_result['validation_errors'].append(f"Missing required columns: {missing_cols}")
    
    for col in df.columns:
        nan_ratio = df[col].isna().sum() / len(df)
        if nan_ratio > allow_nan_ratio:
            validation_result['high_nan_columns'].append({
                'column': col,
                'nan_ratio': nan_ratio
            })
            validation_result['is_valid'] = False
    
    return validation_resultimport pandas as pd

def clean_dataframe(df):
    """
    Clean a pandas DataFrame by removing null values and duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    cleaned_df = df.copy()
    
    cleaned_df = cleaned_df.dropna()
    
    cleaned_df = cleaned_df.drop_duplicates()
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and required columns.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    bool: True if validation passes.
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    if df.empty:
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'name': ['Alice', 'Bob', 'Charlie', None, 'Alice'],
        'age': [25, 30, 35, 40, 25],
        'city': ['NYC', 'LA', 'Chicago', None, 'NYC']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned = clean_dataframe(df)
    print(cleaned)
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Parameters:
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

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return True

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a specific column using IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    multiplier (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        print(f"Error: Column '{column}' not found in DataFrame")
        return df
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    initial_count = len(df)
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    removed_count = initial_count - len(filtered_df)
    
    if removed_count > 0:
        print(f"Removed {removed_count} outliers from column '{column}'")
    
    return filtered_df
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
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

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
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
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
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

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
        return pd.Series([0] * len(data), index=data.index)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process (default: all numeric)
        outlier_method: 'iqr' or 'zscore' (default: 'iqr')
        normalize_method: 'minmax' or 'zscore' (default: 'minmax')
    
    Returns:
        Cleaned DataFrame
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
        else:
            raise ValueError(f"Unknown outlier method: {outlier_method}")
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        if normalize_method == 'minmax':
            cleaned_data[f"{column}_normalized"] = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data[f"{column}_standardized"] = normalize_zscore(cleaned_data, column)
        else:
            raise ValueError(f"Unknown normalize method: {normalize_method}")
    
    return cleaned_data

def validate_data(data, required_columns=None, allow_nan=False):
    """
    Validate dataset structure and content.
    
    Args:
        data: pandas DataFrame to validate
        required_columns: list of required column names
        allow_nan: whether to allow NaN values
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(data, pd.DataFrame):
        return False, "Input must be a pandas DataFrame"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if not allow_nan and data.isnull().any().any():
        nan_columns = data.columns[data.isnull().any()].tolist()
        return False, f"NaN values found in columns: {nan_columns}"
    
    return True, "Data validation passed"

def get_data_summary(data):
    """
    Generate comprehensive data summary.
    
    Args:
        data: pandas DataFrame
    
    Returns:
        Dictionary with data summary
    """
    summary = {
        'shape': data.shape,
        'columns': data.columns.tolist(),
        'dtypes': data.dtypes.to_dict(),
        'missing_values': data.isnull().sum().to_dict(),
        'numeric_summary': {},
        'categorical_summary': {}
    }
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['numeric_summary'][col] = {
            'mean': data[col].mean(),
            'std': data[col].std(),
            'min': data[col].min(),
            '25%': data[col].quantile(0.25),
            '50%': data[col].quantile(0.50),
            '75%': data[col].quantile(0.75),
            'max': data[col].max()
        }
    
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        summary['categorical_summary'][col] = {
            'unique_count': data[col].nunique(),
            'top_value': data[col].mode().iloc[0] if not data[col].mode().empty else None,
            'top_count': data[col].value_counts().iloc[0] if not data[col].value_counts().empty else 0
        }
    
    return summary
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (bool): Whether to fill missing values.
    fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', or 'zero').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing:
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            if cleaned_df[column].isnull().any():
                if fill_strategy == 'mean':
                    fill_value = cleaned_df[column].mean()
                elif fill_strategy == 'median':
                    fill_value = cleaned_df[column].median()
                elif fill_strategy == 'mode':
                    fill_value = cleaned_df[column].mode()[0]
                elif fill_strategy == 'zero':
                    fill_value = 0
                else:
                    raise ValueError("Invalid fill_strategy. Choose from 'mean', 'median', 'mode', or 'zero'.")
                
                cleaned_df[column] = cleaned_df[column].fillna(fill_value)
                print(f"Filled missing values in column '{column}' with {fill_strategy}: {fill_value}")
    
    return cleaned_df

def validate_dataset(df, check_duplicates=True, check_missing=True):
    """
    Validate a DataFrame for duplicates and missing values.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    check_duplicates (bool): Whether to check for duplicates.
    check_missing (bool): Whether to check for missing values.
    
    Returns:
    dict: Validation results.
    """
    validation_results = {}
    
    if check_duplicates:
        duplicate_count = df.duplicated().sum()
        validation_results['duplicate_rows'] = duplicate_count
    
    if check_missing:
        missing_values = df.isnull().sum().sum()
        validation_results['total_missing_values'] = missing_values
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, 5, None, 7],
        'B': [10, 20, 20, None, 50, 60, 70],
        'C': [100, 200, 300, 400, 500, 600, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation before cleaning:")
    print(validate_dataset(df))
    
    cleaned = clean_dataset(df, fill_strategy='median')
    print("\nCleaned DataFrame:")
    print(cleaned)
    print("\nValidation after cleaning:")
    print(validate_dataset(cleaned))
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
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
    Calculate summary statistics for a column after outlier removal.
    
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
        'count': df[column].count()
    }
    
    return stats

def clean_dataset(df, columns_to_clean=None):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns_to_clean is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns_to_clean = numeric_cols
    
    cleaned_df = df.copy()
    
    for column in columns_to_clean:
        if column in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[column]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[::100, 'A'] = 500
    
    print("Original dataset shape:", df.shape)
    print("\nOriginal summary statistics for column 'A':")
    print(calculate_summary_statistics(df, 'A'))
    
    cleaned_df = clean_dataset(df, ['A'])
    
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("\nCleaned summary statistics for column 'A':")
    print(calculate_summary_statistics(cleaned_df, 'A'))
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    outliers_removed = len(data) - len(filtered_data)
    
    return filtered_data, outliers_removed

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    outliers_removed = len(data) - len(filtered_data)
    
    return filtered_data, outliers_removed

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

def clean_dataset(df, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Main function to clean dataset by removing outliers and normalizing numeric columns
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    stats_report = {}
    
    for col in numeric_columns:
        if col not in df.columns:
            continue
            
        original_count = len(cleaned_df)
        
        if outlier_method == 'iqr':
            cleaned_df, outliers = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df, outliers = remove_outliers_zscore(cleaned_df, col)
        else:
            outliers = 0
        
        if normalize_method == 'minmax':
            cleaned_df[f'{col}_normalized'] = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df[f'{col}_standardized'] = normalize_zscore(cleaned_df, col)
        
        stats_report[col] = {
            'original_rows': original_count,
            'cleaned_rows': len(cleaned_df),
            'outliers_removed': outliers,
            'outlier_percentage': (outliers / original_count * 100) if original_count > 0 else 0
        }
    
    return cleaned_df, stats_report

def validate_data(df, required_columns=None, allow_nan_columns=None):
    """
    Validate dataset structure and content
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'empty_columns': [],
        'high_nan_columns': []
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_results['missing_columns'] = missing
            validation_results['is_valid'] = False
    
    for col in df.columns:
        if df[col].isnull().all():
            validation_results['empty_columns'].append(col)
            validation_results['is_valid'] = False
        
        nan_percentage = df[col].isnull().sum() / len(df) * 100
        if nan_percentage > 50 and (allow_nan_columns is None or col not in allow_nan_columns):
            validation_results['high_nan_columns'].append((col, nan_percentage))
    
    return validation_results