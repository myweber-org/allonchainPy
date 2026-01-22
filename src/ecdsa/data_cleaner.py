import pandas as pd
import numpy as np

def remove_duplicates(df):
    """
    Remove duplicate rows from a DataFrame.
    """
    return df.drop_duplicates()

def fill_missing_values(df, strategy='mean'):
    """
    Fill missing values in numeric columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if strategy == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == 'median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df

def remove_outliers(df, column, threshold=3):
    """
    Remove outliers using z-score method.
    """
    if column in df.columns:
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        df = df[z_scores < threshold]
    return df

def standardize_column_names(df):
    """
    Standardize column names to lowercase with underscores.
    """
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df

def clean_dataframe(df, remove_dups=True, fill_na=True, outlier_cols=None):
    """
    Perform a series of cleaning operations on a DataFrame.
    """
    if remove_dups:
        df = remove_duplicates(df)
    if fill_na:
        df = fill_missing_values(df)
    if outlier_cols:
        for col in outlier_cols:
            df = remove_outliers(df, col)
    df = standardize_column_names(df)
    return df