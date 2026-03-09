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
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path, output_path):
    data = pd.read_csv(file_path)
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        data = remove_outliers_iqr(data, col)
    
    for col in numeric_columns:
        data = normalize_minmax(data, col)
    
    data.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    return data

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    cleaned_data = clean_dataset(input_file, output_file)