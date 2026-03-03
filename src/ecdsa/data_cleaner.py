
def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    series = data[column].dropna()
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return outliers.index.tolist()

def remove_outliers(data, column, threshold=1.5):
    """
    Remove outliers from specified column
    """
    outlier_indices = detect_outliers_iqr(data, column, threshold)
    return data.drop(index=outlier_indices)

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    series = data[column].dropna()
    min_val = series.min()
    max_val = series.max()
    
    if max_val == min_val:
        return series
    
    normalized = (series - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    series = data[column].dropna()
    mean_val = series.mean()
    std_val = series.std()
    
    if std_val == 0:
        return series
    
    standardized = (series - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy
    """
    if columns is None:
        columns = data.columns
    
    data_copy = data.copy()
    
    for column in columns:
        if column not in data_copy.columns:
            continue
            
        if data_copy[column].isnull().any():
            if strategy == 'mean':
                fill_value = data_copy[column].mean()
            elif strategy == 'median':
                fill_value = data_copy[column].median()
            elif strategy == 'mode':
                fill_value = data_copy[column].mode()[0]
            elif strategy == 'drop':
                data_copy = data_copy.dropna(subset=[column])
                continue
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            data_copy[column] = data_copy[column].fillna(fill_value)
    
    return data_copy

def clean_dataset(data, config):
    """
    Comprehensive data cleaning pipeline
    """
    cleaned_data = data.copy()
    
    # Handle missing values
    if 'missing_strategy' in config:
        cleaned_data = handle_missing_values(
            cleaned_data, 
            strategy=config['missing_strategy'],
            columns=config.get('columns')
        )
    
    # Remove outliers
    if 'remove_outliers' in config and config['remove_outliers']:
        for column in config.get('outlier_columns', []):
            if column in cleaned_data.columns:
                cleaned_data = remove_outliers(cleaned_data, column)
    
    # Normalize/Standardize
    if 'normalization' in config:
        for column, method in config['normalization'].items():
            if column in cleaned_data.columns:
                if method == 'minmax':
                    cleaned_data[column] = normalize_minmax(cleaned_data, column)
                elif method == 'zscore':
                    cleaned_data[column] = standardize_zscore(cleaned_data, column)
    
    return cleaned_data
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
            cleaned_df[column] = cleaned_df[column].clip(lower=lower_bound, upper=upper_bound)
    
    elif outlier_method == 'zscore':
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            z_scores = np.abs((cleaned_df[column] - cleaned_df[column].mean()) / cleaned_df[column].std())
            cleaned_df = cleaned_df[z_scores < 3]
    
    return cleaned_df

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from dataframe.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    subset (list): Columns to consider for duplicates
    keep (str): Which duplicates to keep ('first', 'last', False)
    
    Returns:
    pd.DataFrame: Dataframe without duplicates
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def normalize_columns(df, columns=None, method='minmax'):
    """
    Normalize specified columns in dataframe.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): Columns to normalize (None for all numeric columns)
    method (str): Normalization method ('minmax', 'standard')
    
    Returns:
    pd.DataFrame: Dataframe with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    normalized_df = df.copy()
    
    if method == 'minmax':
        for col in columns:
            if col in df.columns:
                min_val = normalized_df[col].min()
                max_val = normalized_df[col].max()
                if max_val != min_val:
                    normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
    
    elif method == 'standard':
        for col in columns:
            if col in df.columns:
                mean_val = normalized_df[col].mean()
                std_val = normalized_df[col].std()
                if std_val != 0:
                    normalized_df[col] = (normalized_df[col] - mean_val) / std_val
    
    return normalized_df

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 9],
        'C': [10, 11, 12, 13, 14]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    # Clean the data
    cleaned = clean_dataset(df, missing_strategy='mean', outlier_method='iqr')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    # Normalize the data
    normalized = normalize_columns(cleaned, method='minmax')
    print("\nNormalized DataFrame:")
    print(normalized)
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
    filtered_data = data[(z_scores < threshold) | (z_scores.isna())]
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
        return pd.Series([0] * len(data), index=data.index)
    
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
        return pd.Series([0] * len(data), index=data.index)
    
    return (data[column] - mean_val) / std_val

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='zscore'):
    """
    Clean dataset by removing outliers and normalizing numeric columns.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric column names (default: all numeric columns)
        outlier_method: 'iqr' or 'zscore' (default: 'iqr')
        normalize_method: 'minmax' or 'zscore' (default: 'zscore')
    
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
            cleaned_data[column] = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data[column] = normalize_zscore(cleaned_data, column)
        else:
            raise ValueError(f"Unknown normalize method: {normalize_method}")
    
    return cleaned_data

def validate_data(data, required_columns=None, check_missing=True, check_duplicates=True):
    """
    Validate dataset for common data quality issues.
    
    Args:
        data: pandas DataFrame
        required_columns: list of required column names
        check_missing: check for missing values
        check_duplicates: check for duplicate rows
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {}
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        validation_results['missing_columns'] = missing_columns
    
    if check_missing:
        missing_counts = data.isnull().sum()
        missing_percentage = (missing_counts / len(data)) * 100
        validation_results['missing_values'] = missing_counts[missing_counts > 0].to_dict()
        validation_results['missing_percentage'] = missing_percentage[missing_counts > 0].to_dict()
    
    if check_duplicates:
        duplicate_count = data.duplicated().sum()
        validation_results['duplicate_rows'] = duplicate_count
    
    validation_results['total_rows'] = len(data)
    validation_results['total_columns'] = len(data.columns)
    
    return validation_results
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, columns):
    """Remove outliers using IQR method for specified columns."""
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def remove_outliers_zscore(df, columns, threshold=3):
    """Remove outliers using Z-score method."""
    df_clean = df.copy()
    for col in columns:
        z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
        df_clean = df_clean[z_scores < threshold]
    return df_clean

def normalize_minmax(df, columns):
    """Normalize specified columns using Min-Max scaling."""
    df_norm = df.copy()
    for col in columns:
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
    return df_norm

def normalize_zscore(df, columns):
    """Normalize specified columns using Z-score standardization."""
    df_norm = df.copy()
    for col in columns:
        mean_val = df_norm[col].mean()
        std_val = df_norm[col].std()
        df_norm[col] = (df_norm[col] - mean_val) / std_val
    return df_norm

def handle_missing_values(df, strategy='mean'):
    """Handle missing values using specified strategy."""
    df_filled = df.copy()
    numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        for col in numeric_cols:
            df_filled[col].fillna(df_filled[col].mean(), inplace=True)
    elif strategy == 'median':
        for col in numeric_cols:
            df_filled[col].fillna(df_filled[col].median(), inplace=True)
    elif strategy == 'mode':
        for col in numeric_cols:
            df_filled[col].fillna(df_filled[col].mode()[0], inplace=True)
    elif strategy == 'drop':
        df_filled.dropna(inplace=True)
    
    return df_filled

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax', missing_strategy='mean'):
    """Complete data cleaning pipeline."""
    print(f"Original dataset shape: {df.shape}")
    
    df_clean = handle_missing_values(df, strategy=missing_strategy)
    print(f"After handling missing values: {df_clean.shape}")
    
    if outlier_method == 'iqr':
        df_clean = remove_outliers_iqr(df_clean, numeric_columns)
    elif outlier_method == 'zscore':
        df_clean = remove_outliers_zscore(df_clean, numeric_columns)
    print(f"After outlier removal: {df_clean.shape}")
    
    if normalize_method == 'minmax':
        df_clean = normalize_minmax(df_clean, numeric_columns)
    elif normalize_method == 'zscore':
        df_clean = normalize_zscore(df_clean, numeric_columns)
    
    print(f"Final cleaned dataset shape: {df_clean.shape}")
    return df_clean

def save_cleaned_data(df, output_path):
    """Save cleaned dataset to CSV."""
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    numeric_cols = ['age', 'income', 'score', 'height', 'weight']
    
    try:
        raw_data = load_dataset(input_file)
        cleaned_data = clean_dataset(
            raw_data, 
            numeric_columns=numeric_cols,
            outlier_method='iqr',
            normalize_method='zscore',
            missing_strategy='median'
        )
        save_cleaned_data(cleaned_data, output_file)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")