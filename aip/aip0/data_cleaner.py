
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def clean_dataset(df, numeric_columns):
    original_shape = df.shape
    for col in numeric_columns:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)
    cleaned_shape = df.shape
    removed_count = original_shape[0] - cleaned_shape[0]
    print(f"Removed {removed_count} outliers from dataset")
    return df

def main():
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 100, 27, 28, 29, -10],
        'humidity': [45, 46, 47, 48, 49, 200, 50, 51, 52, -5],
        'pressure': [1013, 1014, 1015, 1016, 1017, 2000, 1018, 1019, 1020, 500]
    }
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    
    numeric_cols = ['temperature', 'humidity', 'pressure']
    cleaned_df = clean_dataset(df, numeric_cols)
    
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = main()