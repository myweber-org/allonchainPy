import pandas as pd
import numpy as np

def clean_csv_data(filepath, strategy='mean', fill_value=None):
    """
    Clean a CSV file by handling missing values.
    
    Args:
        filepath (str): Path to the CSV file.
        strategy (str): Strategy for handling missing values.
            Options: 'mean', 'median', 'mode', 'constant', 'drop'.
        fill_value: Value to use when strategy is 'constant'.
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if df.empty:
        return df
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        df_cleaned = df.dropna()
    elif strategy == 'mean':
        df_cleaned = df.copy()
        for col in numeric_cols:
            df_cleaned[col].fillna(df[col].mean(), inplace=True)
    elif strategy == 'median':
        df_cleaned = df.copy()
        for col in numeric_cols:
            df_cleaned[col].fillna(df[col].median(), inplace=True)
    elif strategy == 'mode':
        df_cleaned = df.copy()
        for col in df.columns:
            mode_val = df[col].mode()
            if not mode_val.empty:
                df_cleaned[col].fillna(mode_val[0], inplace=True)
    elif strategy == 'constant':
        if fill_value is None:
            raise ValueError("fill_value must be provided for constant strategy")
        df_cleaned = df.fillna(fill_value)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return df_cleaned

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to CSV.
    
    Args:
        df (pandas.DataFrame): DataFrame to save.
        output_path (str): Path for output CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    input_file = "sample_data.csv"
    output_file = "cleaned_data.csv"
    
    try:
        cleaned_df = clean_csv_data(input_file, strategy='median')
        save_cleaned_data(cleaned_df, output_file)
        print(f"Original shape: {pd.read_csv(input_file).shape}")
        print(f"Cleaned shape: {cleaned_df.shape}")
    except Exception as e:
        print(f"Error during data cleaning: {e}")import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str or dict): Method to fill missing values. 
                                Can be 'mean', 'median', 'mode', or a dictionary of column:value pairs.
                                If None, missing values are not filled.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        elif fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
    
    return cleaned_df

def remove_outliers_iqr(df, columns=None, multiplier=1.5):
    """
    Remove outliers from specified columns using the Interquartile Range (IQR) method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of column names to process. If None, all numeric columns are used.
    multiplier (float): IQR multiplier for outlier detection. Default is 1.5.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns
    
    df_clean = df.copy()
    
    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean

def standardize_columns(df, columns=None):
    """
    Standardize specified columns to have zero mean and unit variance.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of column names to standardize. If None, all numeric columns are used.
    
    Returns:
    pd.DataFrame: DataFrame with standardized columns.
    """
    from sklearn.preprocessing import StandardScaler
    
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns
    
    df_standardized = df.copy()
    scaler = StandardScaler()
    
    df_standardized[columns] = scaler.fit_transform(df_standardized[columns])
    
    return df_standardized, scaler

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 3, 4, 5, 100],
        'B': [10, 20, 20, 30, None, 50, 60],
        'C': ['x', 'y', 'y', 'z', 'z', 'x', 'x']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    no_outliers = remove_outliers_iqr(cleaned, columns=['A'])
    print("\nDataFrame without outliers in column A:")
    print(no_outliers)
    
    standardized, _ = standardize_columns(no_outliers, columns=['A', 'B'])
    print("\nStandardized DataFrame:")
    print(standardized)