import pandas as pd
import numpy as np

def clean_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    strategy (str): Strategy to handle missing values. 
                    Options: 'mean', 'median', 'mode', 'drop'.
    columns (list): List of columns to apply cleaning. If None, applies to all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    if columns is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    if strategy == 'drop':
        df_clean = df_clean.dropna(subset=columns)
    else:
        for col in columns:
            if df_clean[col].isnull().any():
                if strategy == 'mean':
                    fill_value = df_clean[col].mean()
                elif strategy == 'median':
                    fill_value = df_clean[col].median()
                elif strategy == 'mode':
                    fill_value = df_clean[col].mode()[0]
                else:
                    raise ValueError(f"Unsupported strategy: {strategy}")
                
                df_clean[col] = df_clean[col].fillna(fill_value)
    
    return df_clean

def remove_outliers_iqr(df, columns=None, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range (IQR) method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of columns to check for outliers.
    multiplier (float): IQR multiplier for outlier detection.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean

def standardize_columns(df, columns=None):
    """
    Standardize numeric columns to have zero mean and unit variance.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of columns to standardize.
    
    Returns:
    pd.DataFrame: DataFrame with standardized columns.
    """
    from sklearn.preprocessing import StandardScaler
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    scaler = StandardScaler()
    
    df_clean[columns] = scaler.fit_transform(df_clean[columns])
    
    return df_clean

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, 20, 30, np.nan, 50],
        'C': [100, 200, 300, 400, 500],
        'category': ['X', 'Y', 'X', 'Y', 'Z']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_missing_values(df, strategy='mean')
    print("\nDataFrame after cleaning missing values:")
    print(cleaned_df)
    
    standardized_df = standardize_columns(cleaned_df, columns=['A', 'B', 'C'])
    print("\nDataFrame after standardization:")
    print(standardized_df)