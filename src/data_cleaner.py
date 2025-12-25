
import pandas as pd
import numpy as np

def clean_csv_data(file_path, fill_method='mean', drop_threshold=0.5):
    """
    Load and clean CSV data by handling missing values.
    
    Parameters:
    file_path (str): Path to the CSV file.
    fill_method (str): Method to fill missing values ('mean', 'median', 'mode', 'zero').
    drop_threshold (float): Drop columns with missing ratio above this threshold.
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    missing_ratio = df.isnull().sum() / len(df)
    columns_to_drop = missing_ratio[missing_ratio > drop_threshold].index
    df = df.drop(columns=columns_to_drop)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    
    if fill_method == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif fill_method == 'median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif fill_method == 'zero':
        df[numeric_cols] = df[numeric_cols].fillna(0)
    
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')
    
    return df

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a numeric column using IQR method.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame.
    column (str): Column name to process.
    
    Returns:
    pandas.DataFrame: DataFrame with outliers removed.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def standardize_columns(df, columns):
    """
    Standardize specified numeric columns to zero mean and unit variance.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame.
    columns (list): List of column names to standardize.
    
    Returns:
    pandas.DataFrame: DataFrame with standardized columns.
    """
    df_standardized = df.copy()
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df_standardized[col] = (df[col] - mean) / std
    return df_standardized

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, np.nan, np.nan, 10, 11],
        'C': ['x', 'y', np.nan, 'z', 'w'],
        'D': [100, 200, 300, 400, 500]
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned = clean_csv_data('sample_data.csv', fill_method='median', drop_threshold=0.6)
    print("Cleaned DataFrame:")
    print(cleaned)
    
    cleaned_no_outliers = remove_outliers_iqr(cleaned, 'D')
    print("\nDataFrame after outlier removal:")
    print(cleaned_no_outliers)
    
    standardized = standardize_columns(cleaned_no_outliers, ['D'])
    print("\nDataFrame after standardization:")
    print(standardized)