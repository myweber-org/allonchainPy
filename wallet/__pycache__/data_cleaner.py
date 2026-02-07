import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val == min_val:
        return data[column]
    return (data[column] - min_val) / (max_val - min_val)

def standardize_zscore(data, column):
    mean_val = data[column].mean()
    std_val = data[column].std()
    if std_val == 0:
        return data[column]
    return (data[column] - mean_val) / std_val

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col + '_normalized'] = normalize_minmax(cleaned_df, col)
            cleaned_df[col + '_standardized'] = standardize_zscore(cleaned_df, col)
    return cleaned_df

def generate_summary(cleaned_df, original_df, numeric_columns):
    summary = {}
    for col in numeric_columns:
        if col in cleaned_df.columns:
            summary[col] = {
                'original_count': original_df[col].count(),
                'cleaned_count': cleaned_df[col].count(),
                'removed_outliers': original_df[col].count() - cleaned_df[col].count(),
                'original_mean': original_df[col].mean(),
                'cleaned_mean': cleaned_df[col].mean(),
                'original_std': original_df[col].std(),
                'cleaned_std': cleaned_df[col].std()
            }
    return pd.DataFrame(summary).T

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 200, 1000)
    })
    
    sample_data.loc[np.random.choice(1000, 50), 'feature_a'] = np.random.uniform(300, 500, 50)
    
    numeric_cols = ['feature_a', 'feature_b', 'feature_c']
    cleaned_data = clean_dataset(sample_data, numeric_cols)
    stats_summary = generate_summary(cleaned_data, sample_data, numeric_cols)
    
    print("Original dataset shape:", sample_data.shape)
    print("Cleaned dataset shape:", cleaned_data.shape)
    print("\nStatistical Summary:")
    print(stats_summary)