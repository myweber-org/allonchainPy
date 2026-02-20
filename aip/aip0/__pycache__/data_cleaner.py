
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

def calculate_summary_stats(df, column):
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
    dict: Dictionary of summary statistics for each cleaned column
    """
    if columns_to_clean is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns_to_clean = list(numeric_cols)
    
    cleaned_df = df.copy()
    stats_dict = {}
    
    for column in columns_to_clean:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            
            stats = calculate_summary_stats(cleaned_df, column)
            stats['outliers_removed'] = removed_count
            stats_dict[column] = stats
    
    return cleaned_df, stats_dict

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[::100, 'A'] = 500
    
    print("Original DataFrame shape:", df.shape)
    print("\nOriginal summary statistics:")
    print(df.describe())
    
    cleaned_df, stats = clean_dataset(df, ['A', 'B'])
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("\nCleaned summary statistics:")
    print(cleaned_df.describe())
    
    print("\nDetailed statistics per column:")
    for col, col_stats in stats.items():
        print(f"\n{col}:")
        for stat_name, stat_value in col_stats.items():
            print(f"  {stat_name}: {stat_value:.2f}")
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column):
    """
    Remove outliers using the Interquartile Range method.
    Returns filtered DataFrame and outlier indices.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    return filtered_data, outliers.index.tolist()

def normalize_minmax(data, column):
    """
    Normalize column values to range [0, 1] using min-max scaling.
    Returns new Series with normalized values.
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
    Standardize column values using z-score normalization.
    Returns new Series with standardized values.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values in numeric columns.
    Supported strategies: 'mean', 'median', 'mode', 'drop'
    Returns DataFrame with handled missing values.
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        return data.dropna(subset=numeric_cols)
    
    for col in numeric_cols:
        if data[col].isnull().any():
            if strategy == 'mean':
                fill_value = data[col].mean()
            elif strategy == 'median':
                fill_value = data[col].median()
            elif strategy == 'mode':
                fill_value = data[col].mode()[0]
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")
            
            data[col] = data[col].fillna(fill_value)
    
    return data

def validate_dataframe(data):
    """
    Basic DataFrame validation.
    Returns tuple of (is_valid, issues_list)
    """
    issues = []
    
    if not isinstance(data, pd.DataFrame):
        issues.append("Input is not a pandas DataFrame")
        return False, issues
    
    if data.empty:
        issues.append("DataFrame is empty")
        return False, issues
    
    if data.isnull().all().any():
        issues.append("Some columns contain only null values")
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        issues.append("No numeric columns found")
    
    return len(issues) == 0, issues

def create_sample_dataset():
    """
    Create a sample dataset for testing purposes.
    Returns DataFrame with sample data containing outliers and missing values.
    """
    np.random.seed(42)
    
    data = {
        'feature_a': np.concatenate([
            np.random.normal(50, 10, 90),
            np.array([150, -30, 200])
        ]),
        'feature_b': np.concatenate([
            np.random.normal(100, 20, 93),
            np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        ]),
        'feature_c': np.random.uniform(0, 1, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(data)
    df.loc[95:99, 'feature_a'] = np.nan
    
    return df

if __name__ == "__main__":
    sample_data = create_sample_dataset()
    print("Sample dataset created:")
    print(sample_data.head())
    print(f"\nDataset shape: {sample_data.shape}")
    
    is_valid, issues = validate_dataframe(sample_data)
    print(f"\nData validation: {'Valid' if is_valid else 'Invalid'}")
    if issues:
        print("Issues found:", issues)
    
    cleaned_data, outliers = remove_outliers_iqr(sample_data, 'feature_a')
    print(f"\nOutliers removed from 'feature_a': {len(outliers)} rows")
    
    normalized = normalize_minmax(cleaned_data, 'feature_c')
    print(f"\n'feature_c' normalized to range [0, 1]")
    
    standardized = standardize_zscore(cleaned_data, 'feature_a')
    print(f"'feature_a' standardized using z-score")
    
    filled_data = handle_missing_values(sample_data, strategy='mean')
    print(f"\nMissing values handled using mean imputation")
    print(f"Original null count: {sample_data.isnull().sum().sum()}")
    print(f"Cleaned null count: {filled_data.isnull().sum().sum()}")