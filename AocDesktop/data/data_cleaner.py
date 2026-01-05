import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def normalize_minmax(data, column):
    """
    Normalize column values to range [0, 1] using min-max scaling.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if min_val == max_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def clean_dataset(df, numeric_columns, outlier_factor=1.5):
    """
    Clean dataset by removing outliers and normalizing numeric columns.
    """
    original_shape = df.shape
    
    for col in numeric_columns:
        if col in df.columns:
            df, removed = remove_outliers_iqr(df, col, outlier_factor)
            print(f"Removed {removed} outliers from column '{col}'")
    
    for col in numeric_columns:
        if col in df.columns:
            df[f"{col}_normalized"] = normalize_minmax(df, col)
    
    print(f"Original dataset shape: {original_shape}")
    print(f"Cleaned dataset shape: {df.shape}")
    
    return df

def generate_sample_data():
    """
    Generate sample data for testing.
    """
    np.random.seed(42)
    data = {
        'id': range(100),
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(data)
    df.loc[10:15, 'feature_a'] = 500
    df.loc[20:25, 'feature_b'] = 1000
    
    return df

if __name__ == "__main__":
    sample_df = generate_sample_data()
    numeric_cols = ['feature_a', 'feature_b']
    
    cleaned_df = clean_dataset(sample_df, numeric_cols)
    
    print("\nFirst 5 rows of cleaned dataset:")
    print(cleaned_df.head())
    
    print("\nSummary statistics:")
    print(cleaned_df[numeric_cols].describe())