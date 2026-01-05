
import pandas as pd
import numpy as np

def remove_duplicates(df):
    return df.drop_duplicates()

def fill_missing_values(df, strategy='mean'):
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    else:
        return df.fillna(0)

def normalize_column(df, column_name):
    if column_name in df.columns:
        col = df[column_name]
        df[column_name] = (col - col.min()) / (col.max() - col.min())
    return df

def remove_outliers(df, column_name, threshold=3):
    if column_name in df.columns:
        z_scores = np.abs((df[column_name] - df[column_name].mean()) / df[column_name].std())
        df = df[z_scores < threshold]
    return df

def clean_dataframe(df, operations):
    for operation in operations:
        if operation['type'] == 'remove_duplicates':
            df = remove_duplicates(df)
        elif operation['type'] == 'fill_missing':
            df = fill_missing_values(df, operation.get('strategy', 'mean'))
        elif operation['type'] == 'normalize':
            df = normalize_column(df, operation['column'])
        elif operation['type'] == 'remove_outliers':
            df = remove_outliers(df, operation['column'], operation.get('threshold', 3))
    return df