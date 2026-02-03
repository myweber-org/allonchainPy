
import pandas as pd
import numpy as np

def clean_missing_data(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame using specified strategy.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    columns (list): List of columns to apply cleaning to, None for all columns
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if df.empty:
        return df
    
    if columns is None:
        columns = df.columns
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if df_clean[col].isnull().sum() == 0:
            continue
        
        if strategy == 'mean':
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
        
        elif strategy == 'median':
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        elif strategy == 'mode':
            mode_value = df_clean[col].mode()
            if not mode_value.empty:
                df_clean[col].fillna(mode_value[0], inplace=True)
        
        elif strategy == 'drop':
            df_clean = df_clean.dropna(subset=[col])
    
    return df_clean

def remove_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Remove outliers from specified columns using IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of columns to check for outliers
    method (str): Method for outlier detection ('iqr')
    threshold (float): Threshold multiplier for IQR
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if df.empty or columns is None:
        return df
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if not pd.api.types.is_numeric_dtype(df_clean[col]):
            continue
        
        if method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean

def standardize_columns(df, columns=None):
    """
    Standardize specified columns to have zero mean and unit variance.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of columns to standardize
    
    Returns:
    pd.DataFrame: DataFrame with standardized columns
    """
    if df.empty or columns is None:
        return df
    
    df_standardized = df.copy()
    
    for col in columns:
        if col not in df_standardized.columns:
            continue
            
        if not pd.api.types.is_numeric_dtype(df_standardized[col]):
            continue
        
        mean_val = df_standardized[col].mean()
        std_val = df_standardized[col].std()
        
        if std_val > 0:
            df_standardized[col] = (df_standardized[col] - mean_val) / std_val
    
    return df_standardized

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"
import pandas as pd
import numpy as np

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

def clean_dataset(file_path, numeric_columns):
    df = pd.read_csv(file_path)
    
    for column in numeric_columns:
        if column in df.columns:
            df = remove_outliers_iqr(df, column)
            df = normalize_minmax(df, column)
    
    return df

if __name__ == "__main__":
    cleaned_data = clean_dataset('raw_data.csv', ['age', 'income', 'score'])
    cleaned_data.to_csv('cleaned_data.csv', index=False)
    print(f"Data cleaned. Original shape: {pd.read_csv('raw_data.csv').shape}, Cleaned shape: {cleaned_data.shape}")
import numpy as np
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
    if max_val == min_val:
        return df[column]
    return (df[column] - min_val) / (max_val - min_val)

def standardize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    if std_val == 0:
        return df[column]
    return (df[column] - mean_val) / std_val

def clean_dataset(df, numeric_columns, outlier_removal=True, normalization='standard'):
    cleaned_df = df.copy()
    
    if outlier_removal:
        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            if normalization == 'minmax':
                cleaned_df[col] = normalize_minmax(cleaned_df, col)
            elif normalization == 'standard':
                cleaned_df[col] = standardize_zscore(cleaned_df, col)
    
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    return True
import pandas as pd
import numpy as np

def remove_missing_rows(df, columns=None):
    """
    Remove rows with missing values from DataFrame.
    If columns specified, only consider missing values in those columns.
    """
    if columns:
        return df.dropna(subset=columns)
    return df.dropna()

def fill_missing_with_mean(df, columns):
    """
    Fill missing values in specified columns with column mean.
    """
    df_filled = df.copy()
    for col in columns:
        if col in df.columns:
            mean_val = df[col].mean()
            df_filled[col] = df[col].fillna(mean_val)
    return df_filled

def detect_outliers_iqr(df, column, multiplier=1.5):
    """
    Detect outliers using IQR method.
    Returns boolean Series where True indicates outlier.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def cap_outliers(df, column, method='iqr', multiplier=1.5):
    """
    Cap outliers to specified bounds.
    Supports 'iqr' method for outlier detection.
    """
    df_capped = df.copy()
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        df_capped[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    
    return df_capped

def normalize_column(df, column, method='minmax'):
    """
    Normalize column values.
    Supports 'minmax' and 'zscore' methods.
    """
    df_normalized = df.copy()
    
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val != min_val:
            df_normalized[column] = (df[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val != 0:
            df_normalized[column] = (df[column] - mean_val) / std_val
    
    return df_normalized

def clean_dataset(df, missing_strategy='remove', outlier_strategy='cap'):
    """
    Apply comprehensive cleaning pipeline to dataset.
    """
    df_clean = df.copy()
    
    # Handle missing values
    if missing_strategy == 'remove':
        df_clean = remove_missing_rows(df_clean)
    elif missing_strategy == 'mean':
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean = fill_missing_with_mean(df_clean, numeric_cols)
    
    # Handle outliers for numeric columns
    if outlier_strategy == 'cap':
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_clean = cap_outliers(df_clean, col)
    
    return df_clean