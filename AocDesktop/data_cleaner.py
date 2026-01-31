
def remove_duplicates_preserve_order(iterable):
    seen = set()
    result = []
    for item in iterable:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np
from scipy import stats

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
    return cleaned_df

def normalize_data(df, columns, method='minmax'):
    normalized_df = df.copy()
    for col in columns:
        if col in normalized_df.columns:
            if method == 'minmax':
                min_val = normalized_df[col].min()
                max_val = normalized_df[col].max()
                if max_val != min_val:
                    normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
                else:
                    normalized_df[col] = 0
            elif method == 'zscore':
                mean_val = normalized_df[col].mean()
                std_val = normalized_df[col].std()
                if std_val != 0:
                    normalized_df[col] = (normalized_df[col] - mean_val) / std_val
                else:
                    normalized_df[col] = 0
    return normalized_df

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    if outlier_method == 'iqr':
        df_cleaned = remove_outliers_iqr(df, numeric_columns)
    else:
        df_cleaned = df.copy()
    
    df_normalized = normalize_data(df_cleaned, numeric_columns, method=normalize_method)
    return df_normalized

if __name__ == "__main__":
    sample_data = {
        'feature1': [10, 12, 12, 13, 12, 50, 11, 12, 100, 12],
        'feature2': [1.2, 1.3, 1.1, 1.4, 1.2, 5.0, 1.1, 1.3, 10.0, 1.2],
        'category': ['A', 'B', 'A', 'B', 'A', 'A', 'B', 'B', 'A', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    numeric_cols = ['feature1', 'feature2']
    
    cleaned_df = clean_dataset(df, numeric_cols, outlier_method='iqr', normalize_method='minmax')
    print("Original dataset shape:", df.shape)
    print("Cleaned dataset shape:", cleaned_df.shape)
    print("\nCleaned dataset head:")
    print(cleaned_df.head())