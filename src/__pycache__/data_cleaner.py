
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range (IQR) method.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to process.
    
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

def clean_dataset(df, numeric_columns=None):
    """
    Clean dataset by removing outliers from specified numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        numeric_columns (list): List of numeric column names to clean.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            original_len = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_len - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{col}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    sample_df = pd.DataFrame(sample_data)
    sample_df.loc[::100, 'A'] = 500
    
    print(f"Original dataset shape: {sample_df.shape}")
    cleaned_df = clean_dataset(sample_df)
    print(f"Cleaned dataset shape: {cleaned_df.shape}")
    print(f"Removed {len(sample_df) - len(cleaned_df)} total outliers")
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using Interquartile Range method
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

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    filtered_indices = np.where(z_scores < threshold)[0]
    
    filtered_data = data.iloc[filtered_indices].copy()
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
        normalized = pd.Series([0.5] * len(data), index=data.index)
    else:
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
        standardized = pd.Series([0] * len(data), index=data.index)
    else:
        standardized = (data[column] - mean_val) / std_val
    
    return standardized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy
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

def clean_dataset(data, outlier_method='iqr', outlier_columns=None, 
                  normalize_method=None, normalize_columns=None,
                  missing_strategy='mean'):
    """
    Comprehensive data cleaning pipeline
    """
    cleaned_data = data.copy()
    report = {}
    
    # Handle missing values
    cleaned_data = handle_missing_values(cleaned_data, strategy=missing_strategy)
    report['missing_values_handled'] = True
    
    # Remove outliers
    if outlier_columns:
        total_removed = 0
        for col in outlier_columns:
            if outlier_method == 'iqr':
                cleaned_data, removed = remove_outliers_iqr(cleaned_data, col)
            elif outlier_method == 'zscore':
                cleaned_data, removed = remove_outliers_zscore(cleaned_data, col)
            total_removed += removed
        
        report['outliers_removed'] = total_removed
        report['outlier_method'] = outlier_method
    
    # Normalize data
    if normalize_method and normalize_columns:
        for col in normalize_columns:
            if normalize_method == 'minmax':
                cleaned_data[f'{col}_normalized'] = normalize_minmax(cleaned_data, col)
            elif normalize_method == 'zscore':
                cleaned_data[f'{col}_standardized'] = normalize_zscore(cleaned_data, col)
        
        report['normalization_applied'] = True
        report['normalization_method'] = normalize_method
    
    report['original_samples'] = len(data)
    report['cleaned_samples'] = len(cleaned_data)
    report['columns_processed'] = list(data.columns)
    
    return cleaned_data, report
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers using IQR method
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

def remove_outliers_zscore(dataframe, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(dataframe[column]))
    filtered_df = dataframe[z_scores < threshold]
    return filtered_df

def normalize_minmax(dataframe, columns=None):
    """
    Normalize data using Min-Max scaling
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns
    
    normalized_df = dataframe.copy()
    for col in columns:
        if col in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[col]):
            min_val = dataframe[col].min()
            max_val = dataframe[col].max()
            if max_val != min_val:
                normalized_df[col] = (dataframe[col] - min_val) / (max_val - min_val)
    
    return normalized_df

def normalize_zscore(dataframe, columns=None):
    """
    Normalize data using Z-score standardization
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns
    
    normalized_df = dataframe.copy()
    for col in columns:
        if col in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[col]):
            mean_val = dataframe[col].mean()
            std_val = dataframe[col].std()
            if std_val != 0:
                normalized_df[col] = (dataframe[col] - mean_val) / std_val
    
    return normalized_df

def clean_dataset(dataframe, outlier_method='iqr', normalize_method='minmax', 
                  outlier_threshold=1.5, normalize_columns=None):
    """
    Main function to clean dataset by removing outliers and normalizing
    """
    cleaned_df = dataframe.copy()
    
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_threshold)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col, outlier_threshold)
    
    if normalize_method == 'minmax':
        cleaned_df = normalize_minmax(cleaned_df, normalize_columns)
    elif normalize_method == 'zscore':
        cleaned_df = normalize_zscore(cleaned_df, normalize_columns)
    
    return cleaned_df

def get_data_summary(dataframe):
    """
    Generate summary statistics for the dataset
    """
    summary = {
        'original_rows': len(dataframe),
        'numeric_columns': list(dataframe.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(dataframe.select_dtypes(include=['object']).columns),
        'missing_values': dataframe.isnull().sum().to_dict(),
        'basic_stats': dataframe.describe().to_dict()
    }
    return summary
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    data (list or np.array): The dataset containing the column to clean.
    column (int): Index of the column to process.
    
    Returns:
    np.array: Data with outliers removed from the specified column.
    """
    data = np.array(data)
    column_data = data[:, column].astype(float)
    
    Q1 = np.percentile(column_data, 25)
    Q3 = np.percentile(column_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (column_data >= lower_bound) & (column_data <= upper_bound)
    cleaned_data = data[mask]
    
    return cleaned_data

def clean_dataset(data, columns_to_clean):
    """
    Clean multiple columns in a dataset by removing outliers.
    
    Parameters:
    data (list or np.array): The dataset to clean.
    columns_to_clean (list): List of column indices to process.
    
    Returns:
    np.array: Dataset with outliers removed from specified columns.
    """
    cleaned_data = np.array(data)
    
    for column in columns_to_clean:
        cleaned_data = remove_outliers_iqr(cleaned_data, column)
    
    return cleaned_data

if __name__ == "__main__":
    sample_data = [
        [1.0, 2.0, 100.0],
        [2.0, 3.0, 101.0],
        [3.0, 4.0, 102.0],
        [4.0, 5.0, 1000.0],
        [5.0, 6.0, 103.0],
        [6.0, 7.0, 104.0],
        [7.0, 8.0, 2000.0]
    ]
    
    print("Original data:")
    print(sample_data)
    
    cleaned = clean_dataset(sample_data, [2])
    
    print("\nCleaned data (outliers removed from column 2):")
    print(cleaned)