
import pandas as pd
import numpy as np
from scipy import stats

def detect_outliers_iqr(data_series, threshold=1.5):
    """
    Detect outliers using Interquartile Range method.
    Returns boolean mask where True indicates an outlier.
    """
    Q1 = data_series.quantile(0.25)
    Q3 = data_series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    return (data_series < lower_bound) | (data_series > upper_bound)

def remove_outliers(df, column_names, method='iqr', threshold=1.5):
    """
    Remove outliers from specified columns in DataFrame.
    Supports 'iqr' and 'zscore' methods.
    """
    df_clean = df.copy()
    
    for col in column_names:
        if col not in df_clean.columns:
            continue
            
        if method == 'iqr':
            outlier_mask = detect_outliers_iqr(df_clean[col], threshold)
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
            outlier_mask = z_scores > threshold
            # Align mask with original index
            outlier_mask = pd.Series(outlier_mask, index=df_clean[col].dropna().index)
            outlier_mask = outlier_mask.reindex(df_clean.index).fillna(False)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        df_clean = df_clean[~outlier_mask]
    
    return df_clean.reset_index(drop=True)

def normalize_data(df, column_names, method='minmax'):
    """
    Normalize specified columns in DataFrame.
    Supports 'minmax' and 'standard' normalization.
    """
    df_normalized = df.copy()
    
    for col in column_names:
        if col not in df_normalized.columns:
            continue
            
        if method == 'minmax':
            col_min = df_normalized[col].min()
            col_max = df_normalized[col].max()
            if col_max != col_min:
                df_normalized[col] = (df_normalized[col] - col_min) / (col_max - col_min)
            else:
                df_normalized[col] = 0
                
        elif method == 'standard':
            col_mean = df_normalized[col].mean()
            col_std = df_normalized[col].std()
            if col_std != 0:
                df_normalized[col] = (df_normalized[col] - col_mean) / col_std
            else:
                df_normalized[col] = 0
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    return df_normalized

def clean_dataset(df, numeric_columns, outlier_method='iqr', 
                  outlier_threshold=1.5, normalize_method='standard'):
    """
    Complete data cleaning pipeline: outlier removal and normalization.
    """
    # Remove outliers
    df_clean = remove_outliers(df, numeric_columns, outlier_method, outlier_threshold)
    
    # Normalize data
    df_normalized = normalize_data(df_clean, numeric_columns, normalize_method)
    
    return df_normalized

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if len(df) < min_rows:
        raise ValueError(f"DataFrame must have at least {min_rows} rows")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True

# Example usage function
def process_sample_data():
    """
    Demonstrate the data cleaning functions with sample data.
    """
    np.random.seed(42)
    sample_data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'feature_c': np.random.uniform(0, 1, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    # Introduce some outliers
    sample_data['feature_a'][10] = 500
    sample_data['feature_b'][20] = 1000
    
    df = pd.DataFrame(sample_data)
    
    print("Original DataFrame shape:", df.shape)
    print("\nOriginal statistics:")
    print(df[['feature_a', 'feature_b', 'feature_c']].describe())
    
    # Clean the data
    numeric_cols = ['feature_a', 'feature_b', 'feature_c']
    df_clean = clean_dataset(df, numeric_cols)
    
    print("\nCleaned DataFrame shape:", df_clean.shape)
    print("\nCleaned statistics:")
    print(df_clean[['feature_a', 'feature_b', 'feature_c']].describe())
    
    return df_clean

if __name__ == "__main__":
    cleaned_df = process_sample_data()