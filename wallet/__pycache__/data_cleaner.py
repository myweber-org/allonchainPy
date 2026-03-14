
import pandas as pd
import numpy as np

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
    if max_val == min_val:
        return df[column]
    return (df[column] - min_val) / (max_val - min_val)

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.dropna()

if __name__ == "__main__":
    sample_data = {
        'feature1': [10, 20, 30, 40, 50, 200],
        'feature2': [1, 2, 3, 4, 5, 100],
        'category': ['A', 'B', 'A', 'B', 'A', 'B']
    }
    df = pd.DataFrame(sample_data)
    numeric_cols = ['feature1', 'feature2']
    result = clean_dataset(df, numeric_cols)
    print("Original shape:", df.shape)
    print("Cleaned shape:", result.shape)
    print("Cleaned data:\n", result)