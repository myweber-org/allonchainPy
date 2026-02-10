import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def z_score_normalization(data, column):
    """
    Normalize data using z-score method
    """
    mean = data[column].mean()
    std = data[column].std()
    data[f'{column}_normalized'] = (data[column] - mean) / std
    return data

def min_max_scaling(data, column):
    """
    Scale data to [0,1] range
    """
    min_val = data[column].min()
    max_val = data[column].max()
    data[f'{column}_scaled'] = (data[column] - min_val) / (max_val - min_val)
    return data

def clean_dataset(df, numeric_columns):
    """
    Main cleaning pipeline
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = z_score_normalization(cleaned_df, col)
            cleaned_df = min_max_scaling(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, required_columns):
    """
    Validate dataset structure
    """
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True