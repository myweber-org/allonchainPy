
import pandas as pd
import numpy as np

def clean_csv_data(file_path, fill_method='mean'):
    """
    Load and clean CSV data by handling missing values.
    
    Args:
        file_path: Path to the CSV file
        fill_method: Method for filling missing values ('mean', 'median', 'mode', 'zero')
    
    Returns:
        Cleaned DataFrame
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"Original data shape: {df.shape}")
    print(f"Missing values per column:\n{df.isnull().sum()}")
    
    if fill_method == 'mean':
        df = df.fillna(df.mean(numeric_only=True))
    elif fill_method == 'median':
        df = df.fillna(df.median(numeric_only=True))
    elif fill_method == 'mode':
        df = df.fillna(df.mode().iloc[0])
    elif fill_method == 'zero':
        df = df.fillna(0)
    else:
        raise ValueError("Invalid fill_method. Choose from: 'mean', 'median', 'mode', 'zero'")
    
    print(f"Cleaned data shape: {df.shape}")
    print(f"Missing values after cleaning: {df.isnull().sum().sum()}")
    
    return df

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        df: DataFrame
        column: Column name to check for outliers
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    original_len = len(df)
    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    removed_count = original_len - len(df_clean)
    
    print(f"Removed {removed_count} outliers from column '{column}'")
    
    return df_clean

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Args:
        df: DataFrame
        column: Column name to normalize
        method: Normalization method ('minmax' or 'zscore')
    
    Returns:
        DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val != min_val:
            df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
        else:
            df[column + '_normalized'] = 0
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val != 0:
            df[column + '_normalized'] = (df[column] - mean_val) / std_val
        else:
            df[column + '_normalized'] = 0
    else:
        raise ValueError("Invalid method. Choose from: 'minmax', 'zscore'")
    
    return df

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, 20, 30, np.nan, 50, 60],
        'C': [100, 200, 300, 400, 500, 600]
    }
    
    df_sample = pd.DataFrame(sample_data)
    df_sample.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', fill_method='mean')
    cleaned_df = remove_outliers_iqr(cleaned_df, 'A')
    cleaned_df = normalize_column(cleaned_df, 'C', method='minmax')
    
    print("\nFinal cleaned DataFrame:")
    print(cleaned_df)