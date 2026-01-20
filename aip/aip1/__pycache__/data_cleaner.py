import pandas as pd
import numpy as np

def clean_csv_data(file_path, missing_strategy='mean', columns_to_drop=None):
    """
    Load and clean CSV data by handling missing values and optionally dropping columns.
    
    Parameters:
    file_path (str): Path to the CSV file.
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop').
    columns_to_drop (list): List of column names to drop from the dataset.
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading CSV file: {e}")

    if columns_to_drop:
        df = df.drop(columns=columns_to_drop, errors='ignore')

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns

    if missing_strategy == 'mean':
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    elif missing_strategy == 'median':
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    elif missing_strategy == 'mode':
        for col in numeric_columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
    elif missing_strategy == 'drop':
        df = df.dropna(subset=numeric_columns)
    else:
        raise ValueError("Invalid missing_strategy. Choose from 'mean', 'median', 'mode', 'drop'.")

    for col in categorical_columns:
        df[col] = df[col].fillna('Unknown')

    return df

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a specified column using the Interquartile Range (IQR) method.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame.
    column (str): Column name to process.
    multiplier (float): IQR multiplier for outlier detection.
    
    Returns:
    pandas.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, 20, 30, np.nan, 50, 60],
        'C': ['X', 'Y', np.nan, 'Z', 'X', 'Y']
    }
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', missing_strategy='mean', columns_to_drop=None)
    print("Cleaned DataFrame:")
    print(cleaned_df)
    
    filtered_df = remove_outliers_iqr(cleaned_df, 'A')
    print("\nDataFrame after outlier removal in column 'A':")
    print(filtered_df)
import pandas as pd
import numpy as np

def remove_missing_rows(df, threshold=0.5):
    """
    Remove rows with missing values exceeding threshold percentage.
    
    Args:
        df: pandas DataFrame
        threshold: float between 0 and 1
    
    Returns:
        Cleaned DataFrame
    """
    missing_per_row = df.isnull().mean(axis=1)
    return df[missing_per_row <= threshold].copy()

def fill_missing_with_median(df, columns=None):
    """
    Fill missing values with column median.
    
    Args:
        df: pandas DataFrame
        columns: list of column names or None for all numeric columns
    
    Returns:
        DataFrame with filled values
    """
    df_filled = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns:
            median_val = df[col].median()
            df_filled[col] = df[col].fillna(median_val)
    
    return df_filled

def remove_outliers_iqr(df, columns=None, multiplier=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        df: pandas DataFrame
        columns: list of column names or None for all numeric columns
        multiplier: IQR multiplier for outlier detection
    
    Returns:
        DataFrame without outliers
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def standardize_columns(df, columns=None):
    """
    Standardize numeric columns to have zero mean and unit variance.
    
    Args:
        df: pandas DataFrame
        columns: list of column names or None for all numeric columns
    
    Returns:
        DataFrame with standardized columns
    """
    df_standardized = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                df_standardized[col] = (df[col] - mean_val) / std_val
    
    return df_standardized

def clean_dataset(df, missing_threshold=0.3, outlier_multiplier=1.5):
    """
    Complete data cleaning pipeline.
    
    Args:
        df: pandas DataFrame
        missing_threshold: threshold for removing rows with missing values
        outlier_multiplier: IQR multiplier for outlier detection
    
    Returns:
        Cleaned DataFrame
    """
    print(f"Original shape: {df.shape}")
    
    # Step 1: Remove rows with too many missing values
    df_clean = remove_missing_rows(df, threshold=missing_threshold)
    print(f"After removing missing rows: {df_clean.shape}")
    
    # Step 2: Fill remaining missing values
    df_clean = fill_missing_with_median(df_clean)
    
    # Step 3: Remove outliers
    df_clean = remove_outliers_iqr(df_clean, multiplier=outlier_multiplier)
    print(f"After removing outliers: {df_clean.shape}")
    
    # Step 4: Standardize numeric columns
    df_clean = standardize_columns(df_clean)
    
    return df_clean