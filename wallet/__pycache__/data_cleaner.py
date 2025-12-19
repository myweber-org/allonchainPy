
import pandas as pd
import numpy as np
from scipy import stats

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

def clean_dataset(file_path, columns_to_clean):
    df = pd.read_csv(file_path)
    
    for column in columns_to_clean:
        if column in df.columns:
            df = remove_outliers_iqr(df, column)
            df = normalize_minmax(df, column)
    
    cleaned_file_path = file_path.replace('.csv', '_cleaned.csv')
    df.to_csv(cleaned_file_path, index=False)
    return cleaned_file_path

if __name__ == "__main__":
    data_file = "sample_data.csv"
    columns = ["feature1", "feature2", "feature3"]
    
    try:
        result = clean_dataset(data_file, columns)
        print(f"Cleaned data saved to: {result}")
    except FileNotFoundError:
        print(f"Error: File {data_file} not found")
    except Exception as e:
        print(f"An error occurred: {str(e)}")import pandas as pd

def clean_dataset(df):
    """
    Cleans a pandas DataFrame by removing duplicate rows and
    filling missing numeric values with column median.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    
    # Fill missing numeric values with column median
    numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if df_cleaned[col].isnull().any():
            median_val = df_cleaned[col].median()
            df_cleaned[col] = df_cleaned[col].fillna(median_val)
    
    return df_cleaned

def validate_dataframe(df):
    """
    Validates that the input is a pandas DataFrame and not empty.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if df.empty:
        raise ValueError("DataFrame is empty")
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 30, 40, 50],
        'C': ['x', 'y', 'y', 'z', 'x']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned_df = clean_dataset(df)
    print(cleaned_df)
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        multiplier: IQR multiplier for outlier detection
    
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
        return pd.Series([0.5] * len(data), index=data.index)
    
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

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalization_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process
        outlier_method: 'iqr' or None for outlier removal
        normalization_method: 'minmax', 'zscore', or None
    
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
            q1 = cleaned_data[column].quantile(0.25)
            q3 = cleaned_data[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            mask = (cleaned_data[column] >= lower_bound) & (cleaned_data[column] <= upper_bound)
            cleaned_data = cleaned_data[mask]
        
        if normalization_method == 'minmax':
            min_val = cleaned_data[column].min()
            max_val = cleaned_data[column].max()
            if max_val != min_val:
                cleaned_data[column] = (cleaned_data[column] - min_val) / (max_val - min_val)
        
        elif normalization_method == 'zscore':
            mean_val = cleaned_data[column].mean()
            std_val = cleaned_data[column].std()
            if std_val != 0:
                cleaned_data[column] = (cleaned_data[column] - mean_val) / std_val
    
    return cleaned_data

def get_statistical_summary(data, column):
    """
    Get comprehensive statistical summary for a column.
    
    Args:
        data: pandas DataFrame
        column: column name
    
    Returns:
        Dictionary with statistical measures
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    series = data[column]
    summary = {
        'count': len(series),
        'mean': series.mean(),
        'median': series.median(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max(),
        'q1': series.quantile(0.25),
        'q3': series.quantile(0.75),
        'skewness': series.skew(),
        'kurtosis': series.kurtosis()
    }
    
    return summary