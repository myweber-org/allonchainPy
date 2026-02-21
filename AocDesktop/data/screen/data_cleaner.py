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

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def standardize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_standardized'] = (df[column] - mean_val) / std_val
    return df

def handle_missing_values(df, strategy='mean'):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0]
            else:
                fill_value = 0
            df[col].fillna(fill_value, inplace=True)
    return df

def validate_dataframe(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if df.empty:
        raise ValueError("DataFrame is empty")
    return True

def process_numeric_data(df, columns, remove_outliers=True, normalize=True, standardize=False):
    validate_dataframe(df)
    result_df = df.copy()
    
    for col in columns:
        if col not in result_df.columns:
            raise KeyError(f"Column {col} not found in DataFrame")
        if remove_outliers:
            result_df = remove_outliers_iqr(result_df, col)
        if normalize:
            result_df = normalize_minmax(result_df, col)
        if standardize:
            result_df = standardize_zscore(result_df, col)
    
    return result_dfimport pandas as pd
import numpy as np
from scipy import stats

def clean_dataset(df, numeric_columns=None, method='median', z_threshold=3):
    """
    Clean dataset by handling missing values and removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    numeric_columns (list): List of numeric column names to process
    method (str): Imputation method ('mean', 'median', 'mode')
    z_threshold (float): Z-score threshold for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    df_clean = df.copy()
    
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_columns:
        if col not in df.columns:
            continue
            
        col_data = df_clean[col]
        
        if method == 'mean':
            fill_value = col_data.mean()
        elif method == 'median':
            fill_value = col_data.median()
        elif method == 'mode':
            fill_value = col_data.mode()[0] if not col_data.mode().empty else np.nan
        else:
            fill_value = col_data.median()
        
        df_clean[col] = col_data.fillna(fill_value)
        
        z_scores = np.abs(stats.zscore(df_clean[col]))
        outlier_mask = z_scores > z_threshold
        
        if outlier_mask.any():
            median_val = df_clean.loc[~outlier_mask, col].median()
            df_clean.loc[outlier_mask, col] = median_val
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Parameters:
    df (pd.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'null_counts': {},
        'data_types': {}
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_results['missing_columns'] = missing
            validation_results['is_valid'] = False
    
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            validation_results['null_counts'][col] = null_count
        
        validation_results['data_types'][col] = str(df[col].dtype)
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 9],
        'C': ['x', 'y', 'z', 'x', 'y']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation Results:")
    print(validate_dataframe(df, required_columns=['A', 'B', 'C']))
    
    cleaned_df = clean_dataset(df, numeric_columns=['A', 'B'])
    print("\nCleaned DataFrame:")
    print(cleaned_df)