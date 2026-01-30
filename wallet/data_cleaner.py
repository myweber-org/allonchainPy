
import pandas as pd
import numpy as np

def clean_csv_data(filepath, missing_strategy='mean'):
    """
    Load and clean CSV data by handling missing values.
    
    Parameters:
    filepath (str): Path to the CSV file
    missing_strategy (str): Strategy for handling missing values.
                            Options: 'mean', 'median', 'drop', 'zero'
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    
    original_shape = df.shape
    
    if missing_strategy == 'drop':
        df_cleaned = df.dropna()
    elif missing_strategy == 'zero':
        df_cleaned = df.fillna(0)
    elif missing_strategy == 'mean':
        df_cleaned = df.fillna(df.mean(numeric_only=True))
    elif missing_strategy == 'median':
        df_cleaned = df.fillna(df.median(numeric_only=True))
    else:
        raise ValueError(f"Invalid strategy: {missing_strategy}")
    
    cleaned_shape = df_cleaned.shape
    
    print(f"Original data shape: {original_shape}")
    print(f"Cleaned data shape: {cleaned_shape}")
    print(f"Missing values handled using: {missing_strategy} strategy")
    
    return df_cleaned

def detect_outliers_iqr(df, column):
    """
    Detect outliers in a column using IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to check for outliers
    
    Returns:
    pd.Series: Boolean series indicating outliers
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not np.issubdtype(df[column].dtype, np.number):
        raise TypeError(f"Column '{column}' must be numeric")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    outlier_count = outliers.sum()
    if outlier_count > 0:
        print(f"Found {outlier_count} outliers in column '{column}'")
    
    return outliers

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.Series: Normalized column values
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not np.issubdtype(df[column].dtype, np.number):
        raise TypeError(f"Column '{column}' must be numeric")
    
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val == min_val:
            return pd.Series([0.5] * len(df), index=df.index)
        normalized = (df[column] - min_val) / (max_val - min_val)
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val == 0:
            return pd.Series([0] * len(df), index=df.index)
        normalized = (df[column] - mean_val) / std_val
    else:
        raise ValueError(f"Invalid normalization method: {method}")
    
    return normalized

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, 20, 30, np.nan, 50],
        'C': [100, 200, 300, 400, 500]
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', missing_strategy='mean')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    outliers = detect_outliers_iqr(cleaned_df, 'C')
    print(f"\nOutliers in column C: {outliers.sum()}")
    
    normalized_col = normalize_column(cleaned_df, 'C', method='minmax')
    print(f"\nNormalized column C (minmax):")
    print(normalized_col)