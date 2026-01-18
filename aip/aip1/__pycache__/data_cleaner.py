
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
    
    return filtered_df.copy()

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

def process_numerical_data(df, columns):
    """
    Process multiple numerical columns by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to process
    
    Returns:
    pd.DataFrame: Processed DataFrame
    """
    processed_df = df.copy()
    
    for col in columns:
        if col in processed_df.columns and pd.api.types.is_numeric_dtype(processed_df[col]):
            processed_df = remove_outliers_iqr(processed_df, col)
    
    return processed_dfimport numpy as np
import pandas as pd

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

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 200),
        'feature2': np.random.exponential(50, 200),
        'category': np.random.choice(['A', 'B', 'C'], 200)
    })
    
    cleaned = clean_dataset(sample_data, ['feature1', 'feature2'])
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned.shape}")
    print(cleaned.head())
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    outliers_removed = len(data) - len(filtered_data)
    
    return filtered_data, outliers_removed

def zscore_normalize(data, column):
    """
    Normalize data using Z-score normalization.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column]
    
    normalized = (data[column] - mean_val) / std_val
    return normalized

def minmax_normalize(data, column, feature_range=(0, 1)):
    """
    Normalize data using Min-Max scaling.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column]
    
    a, b = feature_range
    normalized = a + ((data[column] - min_val) * (b - a)) / (max_val - min_val)
    return normalized

def detect_missing_patterns(data, threshold=0.3):
    """
    Detect columns with high percentage of missing values.
    """
    missing_percent = data.isnull().sum() / len(data)
    high_missing_cols = missing_percent[missing_percent > threshold].index.tolist()
    
    return {
        'missing_percentages': missing_percent,
        'high_missing_columns': high_missing_cols,
        'total_missing': data.isnull().sum().sum()
    }

def clean_dataset(data, numeric_columns=None, outlier_factor=1.5, normalize_method='zscore'):
    """
    Comprehensive data cleaning pipeline.
    """
    cleaned_data = data.copy()
    
    if numeric_columns is None:
        numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaning_report = {
        'original_shape': data.shape,
        'outliers_removed': {},
        'normalized_columns': []
    }
    
    for col in numeric_columns:
        if col in cleaned_data.columns:
            cleaned_data, outliers = remove_outliers_iqr(cleaned_data, col, outlier_factor)
            cleaning_report['outliers_removed'][col] = outliers
            
            if normalize_method == 'zscore':
                cleaned_data[f'{col}_normalized'] = zscore_normalize(cleaned_data, col)
                cleaning_report['normalized_columns'].append(f'{col}_normalized')
            elif normalize_method == 'minmax':
                cleaned_data[f'{col}_normalized'] = minmax_normalize(cleaned_data, col)
                cleaning_report['normalized_columns'].append(f'{col}_normalized')
    
    cleaning_report['final_shape'] = cleaned_data.shape
    cleaning_report['missing_info'] = detect_missing_patterns(cleaned_data)
    
    return cleaned_data, cleaning_report

def validate_data_types(data, expected_types):
    """
    Validate that columns have expected data types.
    """
    validation_results = {}
    
    for col, expected_type in expected_types.items():
        if col in data.columns:
            actual_type = str(data[col].dtype)
            validation_results[col] = {
                'expected': expected_type,
                'actual': actual_type,
                'valid': expected_type in actual_type
            }
    
    return validation_results

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'age': [25, 30, 35, 200, 28, 32, 150, 29],
        'salary': [50000, 60000, 70000, 1000000, 55000, 65000, 2000000, 58000],
        'score': [85, 90, 88, 30, 92, 87, 15, 89]
    })
    
    print("Original Data:")
    print(sample_data)
    print("\n" + "="*50)
    
    cleaned_data, report = clean_dataset(sample_data, normalize_method='minmax')
    
    print("\nCleaned Data:")
    print(cleaned_data)
    print("\nCleaning Report:")
    for key, value in report.items():
        print(f"{key}: {value}")