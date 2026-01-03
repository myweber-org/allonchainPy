import numpy as np
import pandas as pd

def remove_outliers_iqr(df, columns):
    """
    Remove outliers using the Interquartile Range method.
    """
    cleaned_df = df.copy()
    for col in columns:
        if col in cleaned_df.columns:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    return cleaned_df

def normalize_minmax(df, columns):
    """
    Normalize specified columns using Min-Max scaling.
    """
    normalized_df = df.copy()
    for col in columns:
        if col in normalized_df.columns:
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            if max_val > min_val:
                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
    return normalized_df

def standardize_zscore(df, columns):
    """
    Standardize specified columns using Z-score normalization.
    """
    standardized_df = df.copy()
    for col in columns:
        if col in standardized_df.columns:
            mean_val = standardized_df[col].mean()
            std_val = standardized_df[col].std()
            if std_val > 0:
                standardized_df[col] = (standardized_df[col] - mean_val) / std_val
    return standardized_df

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in specified columns using given strategy.
    """
    filled_df = df.copy()
    if columns is None:
        columns = filled_df.columns
    
    for col in columns:
        if col in filled_df.columns and filled_df[col].isnull().any():
            if strategy == 'mean':
                fill_value = filled_df[col].mean()
            elif strategy == 'median':
                fill_value = filled_df[col].median()
            elif strategy == 'mode':
                fill_value = filled_df[col].mode()[0]
            elif strategy == 'constant':
                fill_value = 0
            else:
                raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'constant'")
            
            filled_df[col].fillna(fill_value, inplace=True)
    
    return filled_df

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method=None, missing_strategy='mean'):
    """
    Comprehensive data cleaning pipeline.
    """
    if outlier_method == 'iqr':
        df_clean = remove_outliers_iqr(df, numeric_columns)
    else:
        df_clean = df.copy()
    
    df_clean = handle_missing_values(df_clean, strategy=missing_strategy, columns=numeric_columns)
    
    if normalize_method == 'minmax':
        df_clean = normalize_minmax(df_clean, numeric_columns)
    elif normalize_method == 'zscore':
        df_clean = standardize_zscore(df_clean, numeric_columns)
    
    return df_clean
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def normalize_minmax(df, columns):
    df_norm = df.copy()
    for col in columns:
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        if max_val != min_val:
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        else:
            df_norm[col] = 0
    return df_norm

def clean_dataset(df, numeric_columns):
    print(f"Original shape: {df.shape}")
    df_no_outliers = remove_outliers_iqr(df, numeric_columns)
    print(f"After outlier removal: {df_no_outliers.shape}")
    df_normalized = normalize_minmax(df_no_outliers, numeric_columns)
    print("Normalization completed")
    return df_normalized

if __name__ == "__main__":
    sample_data = {
        'feature1': np.random.normal(50, 15, 1000),
        'feature2': np.random.exponential(20, 1000),
        'feature3': np.random.uniform(0, 100, 1000)
    }
    df = pd.DataFrame(sample_data)
    df.iloc[::50, 0] = 200
    df.iloc[::30, 1] = 150
    
    cleaned_df = clean_dataset(df, ['feature1', 'feature2', 'feature3'])
    print(f"Cleaned data shape: {cleaned_df.shape}")
    print(cleaned_df.describe())