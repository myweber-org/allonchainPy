import numpy as np
import pandas as pd

def remove_outliers_iqr(df, columns):
    cleaned_df = df.copy()
    for col in columns:
        if col in cleaned_df.columns:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    return cleaned_df.reset_index(drop=True)

def normalize_minmax(df, columns):
    normalized_df = df.copy()
    for col in columns:
        if col in normalized_df.columns:
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            if max_val > min_val:
                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
    return normalized_df

def clean_dataset(df, numeric_columns):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    df_clean = df.dropna(subset=numeric_columns)
    df_clean = remove_outliers_iqr(df_clean, numeric_columns)
    df_clean = normalize_minmax(df_clean, numeric_columns)
    
    return df_clean

def validate_data(df, required_columns):
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 50),
        'feature_b': np.random.exponential(2, 50),
        'feature_c': np.random.uniform(0, 1, 50)
    })
    
    try:
        validate_data(sample_data, ['feature_a', 'feature_b', 'feature_c'])
        cleaned_data = clean_dataset(sample_data, ['feature_a', 'feature_b', 'feature_c'])
        print(f"Original shape: {sample_data.shape}")
        print(f"Cleaned shape: {cleaned_data.shape}")
        print("Data cleaning completed successfully")
    except Exception as e:
        print(f"Error during data cleaning: {e}")