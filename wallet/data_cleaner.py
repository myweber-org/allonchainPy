import numpy as np
import pandas as pd

def remove_outliers_iqr(df, columns, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to process
        factor: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_minmax(df, columns):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to normalize
    
    Returns:
        DataFrame with normalized columns
    """
    df_norm = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        min_val = df[col].min()
        max_val = df[col].max()
        
        if max_val > min_val:
            df_norm[col] = (df[col] - min_val) / (max_val - min_val)
    
    return df_norm

def standardize_zscore(df, columns):
    """
    Standardize data using Z-score normalization.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to standardize
    
    Returns:
        DataFrame with standardized columns
    """
    df_std = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        mean_val = df[col].mean()
        std_val = df[col].std()
        
        if std_val > 0:
            df_std[col] = (df[col] - mean_val) / std_val
    
    return df_std

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        df: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'drop'
        columns: list of columns to process (None for all numeric columns)
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_processed = df.copy()
    
    for col in columns:
        if col not in df.columns or df[col].notna().all():
            continue
        
        if strategy == 'drop':
            df_processed = df_processed.dropna(subset=[col])
        elif strategy == 'mean':
            df_processed[col] = df_processed[col].fillna(df[col].mean())
        elif strategy == 'median':
            df_processed[col] = df_processed[col].fillna(df[col].median())
        elif strategy == 'mode':
            df_processed[col] = df_processed[col].fillna(df[col].mode()[0])
    
    return df_processed.reset_index(drop=True)

def clean_data_pipeline(df, numeric_columns, outlier_factor=1.5):
    """
    Complete data cleaning pipeline.
    
    Args:
        df: pandas DataFrame
        numeric_columns: list of numeric column names
        outlier_factor: IQR multiplier for outlier removal
    
    Returns:
        Cleaned and normalized DataFrame
    """
    # Handle missing values
    df_clean = handle_missing_values(df, strategy='median', columns=numeric_columns)
    
    # Remove outliers
    df_clean = remove_outliers_iqr(df_clean, numeric_columns, factor=outlier_factor)
    
    # Normalize data
    df_clean = normalize_minmax(df_clean, numeric_columns)
    
    return df_clean

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(100, 15, 100),
        'feature2': np.random.exponential(50, 100),
        'feature3': np.random.uniform(0, 1, 100)
    }
    
    # Add some outliers and missing values
    data['feature1'][10] = 500  # Outlier
    data['feature2'][20] = None  # Missing value
    
    df = pd.DataFrame(data)
    
    # Apply cleaning pipeline
    numeric_cols = ['feature1', 'feature2', 'feature3']
    cleaned_df = clean_data_pipeline(df, numeric_cols)
    
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Original stats:\n{df[numeric_cols].describe()}")
    print(f"Cleaned stats:\n{cleaned_df[numeric_cols].describe()}")