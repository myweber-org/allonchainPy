import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns=None, factor=1.5):
    """
    Remove outliers using IQR method.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_minmax(df, columns=None):
    """
    Normalize data using min-max scaling.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    for col in columns:
        min_val = df_normalized[col].min()
        max_val = df_normalized[col].max()
        if max_val != min_val:
            df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
    
    return df_normalized

def clean_dataset(df, outlier_cols=None, normalize_cols=None, outlier_factor=1.5):
    """
    Main cleaning pipeline.
    """
    df_clean = remove_outliers_iqr(df, outlier_cols, outlier_factor)
    df_final = normalize_minmax(df_clean, normalize_cols)
    
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {df_final.shape}")
    print(f"Removed {len(df) - len(df_final)} outliers")
    
    return df_final

def generate_sample_data():
    """
    Generate sample data with outliers for testing.
    """
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(50, 10, 100),
        'feature_b': np.random.exponential(20, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(data)
    df.loc[10, 'feature_a'] = 200
    df.loc[20, 'feature_b'] = 300
    
    return df

if __name__ == "__main__":
    sample_df = generate_sample_data()
    cleaned_df = clean_dataset(
        sample_df,
        outlier_cols=['feature_a', 'feature_b'],
        normalize_cols=['feature_a', 'feature_b']
    )
    
    print("\nFirst 5 rows of cleaned data:")
    print(cleaned_df.head())