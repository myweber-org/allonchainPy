
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def remove_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def normalize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_standardized'] = (df[column] - mean_val) / std_val
    return df

def handle_missing_values(df, strategy='mean'):
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    elif strategy == 'drop':
        return df.dropna()
    else:
        raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'mode', or 'drop'")

def validate_dataframe(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if df.empty:
        raise ValueError("DataFrame is empty")
    return True

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax', missing_strategy='mean'):
    try:
        validate_dataframe(df)
    except (TypeError, ValueError) as e:
        print(f"Data validation failed: {e}")
        return None
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col not in cleaned_df.columns:
            print(f"Column {col} not found in DataFrame")
            continue
            
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        
        if normalize_method == 'minmax':
            cleaned_df = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df = normalize_zscore(cleaned_df, col)
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Rows removed: {df.shape[0] - cleaned_df.shape[0]}")
    
    return cleaned_df
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_duplicates (bool): Whether to remove duplicate rows
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
        print("Removed rows with missing values")
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
    required_columns (list): List of column names that must be present
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

def get_data_summary(df):
    """
    Generate a summary of the DataFrame including missing values and data types.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    dict: Summary statistics
    """
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.astype(str).to_dict(),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns)
    }
    
    if summary['numeric_columns']:
        summary['numeric_stats'] = df[summary['numeric_columns']].describe().to_dict()
    
    return summary

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'name': ['Alice', 'Bob', 'Charlie', None, 'Eve', 'Eve'],
        'age': [25, 30, None, 40, 35, 35],
        'score': [85.5, 92.0, 78.5, None, 88.0, 88.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    print("\nData Summary:")
    summary = get_data_summary(cleaned)
    for key, value in summary.items():
        if key not in ['numeric_stats', 'missing_values']:
            print(f"{key}: {value}")
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

def z_score_normalize(data, column):
    """
    Normalize data using z-score normalization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column]
    
    normalized = (data[column] - mean_val) / std_val
    return normalized

def min_max_normalize(data, column, feature_range=(0, 1)):
    """
    Normalize data using min-max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column]
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    
    if feature_range != (0, 1):
        new_min, new_max = feature_range
        normalized = normalized * (new_max - new_min) + new_min
    
    return normalized

def detect_skewed_columns(data, threshold=0.5):
    """
    Detect columns with skewed distributions
    """
    skewed_columns = []
    
    for column in data.select_dtypes(include=[np.number]).columns:
        skewness = stats.skew(data[column].dropna())
        if abs(skewness) > threshold:
            skewed_columns.append((column, skewness))
    
    return sorted(skewed_columns, key=lambda x: abs(x[1]), reverse=True)

def log_transform(data, column):
    """
    Apply log transformation to reduce skewness
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if data[column].min() <= 0:
        transformed = np.log1p(data[column] - data[column].min() + 1)
    else:
        transformed = np.log(data[column])
    
    return transformed

def clean_dataset(data, numeric_columns=None, outlier_factor=1.5, normalize_method='zscore'):
    """
    Comprehensive data cleaning pipeline
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    cleaning_report = {}
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        original_count = len(cleaned_data)
        
        cleaned_data, removed = remove_outliers_iqr(cleaned_data, column, outlier_factor)
        cleaning_report[column] = {
            'outliers_removed': removed,
            'percentage_removed': (removed / original_count) * 100
        }
        
        if normalize_method == 'zscore':
            cleaned_data[f'{column}_normalized'] = z_score_normalize(cleaned_data, column)
        elif normalize_method == 'minmax':
            cleaned_data[f'{column}_normalized'] = min_max_normalize(cleaned_data, column)
    
    skewed_cols = detect_skewed_columns(cleaned_data[numeric_columns])
    
    for column, skewness in skewed_cols:
        if column in cleaned_data.columns:
            cleaned_data[f'{column}_log'] = log_transform(cleaned_data, column)
    
    return cleaned_data, cleaning_report
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
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
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

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
    
    return (data[column] - min_val) / (max_val - min_val)

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
    
    return (data[column] - mean_val) / std_val

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Clean dataset by removing outliers and normalizing numeric columns
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
        
        if normalize_method == 'minmax':
            cleaned_data[column] = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data[column] = normalize_zscore(cleaned_data, column)
    
    return cleaned_data

def validate_data(data, required_columns=None, check_missing=True, check_duplicates=True):
    """
    Validate dataset for common data quality issues
    """
    validation_results = {}
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        validation_results['missing_columns'] = missing_columns
    
    if check_missing:
        missing_values = data.isnull().sum().to_dict()
        validation_results['missing_values'] = missing_values
    
    if check_duplicates:
        duplicate_count = data.duplicated().sum()
        validation_results['duplicate_rows'] = duplicate_count
    
    return validation_results