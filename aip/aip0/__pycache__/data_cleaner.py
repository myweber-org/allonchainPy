
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

def remove_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def normalize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_zscore'] = (df[column] - mean_val) / std_val
    return df

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            if outlier_method == 'iqr':
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            elif outlier_method == 'zscore':
                cleaned_df = remove_outliers_zscore(cleaned_df, col)
            
            if normalize_method == 'minmax':
                cleaned_df = normalize_minmax(cleaned_df, col)
            elif normalize_method == 'zscore':
                cleaned_df = normalize_zscore(cleaned_df, col)
    
    return cleaned_df

def validate_cleaning(df, original_df, column):
    print(f"Original {column} stats:")
    print(f"  Mean: {original_df[column].mean():.2f}")
    print(f"  Std: {original_df[column].std():.2f}")
    print(f"  Min: {original_df[column].min():.2f}")
    print(f"  Max: {original_df[column].max():.2f}")
    
    print(f"\nCleaned {column} stats:")
    print(f"  Mean: {df[column].mean():.2f}")
    print(f"  Std: {df[column].std():.2f}")
    print(f"  Min: {df[column].min():.2f}")
    print(f"  Max: {df[column].max():.2f}")
    
    if column + '_normalized' in df.columns:
        print(f"\nNormalized {column} stats:")
        print(f"  Min: {df[column + '_normalized'].min():.2f}")
        print(f"  Max: {df[column + '_normalized'].max():.2f}")