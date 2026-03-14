
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
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns
    """
    if columns is None:
        columns = data.columns
    
    data_clean = data.copy()
    
    for col in columns:
        if col not in data.columns:
            continue
            
        if data[col].isnull().any():
            if strategy == 'mean':
                fill_value = data[col].mean()
            elif strategy == 'median':
                fill_value = data[col].median()
            elif strategy == 'mode':
                fill_value = data[col].mode()[0]
            elif strategy == 'drop':
                data_clean = data_clean.dropna(subset=[col])
                continue
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            data_clean[col] = data_clean[col].fillna(fill_value)
    
    return data_clean

def detect_skewness(data, column, threshold=0.5):
    """
    Detect skewness in data column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    skewness = data[column].skew()
    is_skewed = abs(skewness) > threshold
    
    return {
        'skewness': skewness,
        'is_skewed': is_skewed,
        'skew_direction': 'right' if skewness > 0 else 'left' if skewness < 0 else 'none'
    }

def create_clean_data_report(data, numeric_columns=None):
    """
    Generate a cleaning report for the dataset
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns
    
    report = {
        'original_shape': data.shape,
        'missing_values': data.isnull().sum().to_dict(),
        'numeric_columns': list(numeric_columns),
        'column_stats': {}
    }
    
    for col in numeric_columns:
        if col in data.columns:
            report['column_stats'][col] = {
                'mean': data[col].mean(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max(),
                'skewness': data[col].skew(),
                'missing_count': data[col].isnull().sum()
            }
    
    return report
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_method='iqr'):
    """
    Clean a dataset by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    missing_strategy (str): Strategy for missing values ('mean', 'median', 'mode', 'drop')
    outlier_method (str): Method for outlier detection ('iqr', 'zscore')
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean())
    elif missing_strategy == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median())
    elif missing_strategy == 'mode':
        cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
    elif missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    
    # Handle outliers
    if outlier_method == 'iqr':
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            Q1 = cleaned_df[column].quantile(0.25)
            Q3 = cleaned_df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cleaned_df[column] = cleaned_df[column].clip(lower_bound, upper_bound)
    
    elif outlier_method == 'zscore':
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            z_scores = np.abs((cleaned_df[column] - cleaned_df[column].mean()) / cleaned_df[column].std())
            cleaned_df = cleaned_df[z_scores < 3]
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    
    Parameters:
    df (pd.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    bool: True if validation passes
    """
    if len(df) < min_rows:
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 8],
        'C': [10, 20, 30, 40, 50]
    })
    
    print("Original data:")
    print(sample_data)
    
    cleaned = clean_dataset(sample_data, missing_strategy='mean', outlier_method='iqr')
    print("\nCleaned data:")
    print(cleaned)
    
    is_valid = validate_data(cleaned, required_columns=['A', 'B', 'C'], min_rows=3)
    print(f"\nData validation: {is_valid}")