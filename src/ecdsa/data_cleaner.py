import pandas as pd
import numpy as np
from scipy import stats

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

def impute_missing_with_median(data, column):
    median_value = data[column].median()
    data[column].fillna(median_value, inplace=True)
    return data

def remove_duplicates(data):
    return data.drop_duplicates()

def clean_dataset(data, numeric_columns):
    cleaned_data = data.copy()
    cleaned_data = remove_duplicates(cleaned_data)
    
    for col in numeric_columns:
        cleaned_data = impute_missing_with_median(cleaned_data, col)
        outliers = detect_outliers_iqr(cleaned_data, col)
        if not outliers.empty:
            median_val = cleaned_data[col].median()
            cleaned_data.loc[outliers.index, col] = median_val
    
    return cleaned_data

def calculate_summary_statistics(data, columns):
    summary = {}
    for col in columns:
        summary[col] = {
            'mean': data[col].mean(),
            'median': data[col].median(),
            'std': data[col].std(),
            'min': data[col].min(),
            'max': data[col].max()
        }
    return pd.DataFrame(summary)