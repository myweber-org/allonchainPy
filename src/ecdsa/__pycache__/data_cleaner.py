import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas Series using the IQR method.
    Returns a filtered Series.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def z_score_normalize(data, column):
    """
    Normalize a column using Z-score normalization.
    Returns a Series with normalized values.
    """
    mean = data[column].mean()
    std = data[column].std()
    normalized = (data[column] - mean) / std
    return normalized

def min_max_normalize(data, column):
    """
    Normalize a column using Min-Max scaling to range [0, 1].
    Returns a Series with normalized values.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def clean_dataset(df, numeric_columns, outlier_removal=True, normalization='zscore'):
    """
    Main cleaning function to process numeric columns.
    Options for outlier removal and normalization type.
    Returns a cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if outlier_removal:
        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            if normalization == 'zscore':
                cleaned_df[col + '_normalized'] = z_score_normalize(cleaned_df, col)
            elif normalization == 'minmax':
                cleaned_df[col + '_normalized'] = min_max_normalize(cleaned_df, col)
    
    return cleaned_df

def summary_statistics(df, numeric_columns):
    """
    Calculate summary statistics for specified numeric columns.
    Returns a DataFrame with statistics.
    """
    stats_dict = {
        'mean': df[numeric_columns].mean(),
        'median': df[numeric_columns].median(),
        'std': df[numeric_columns].std(),
        'min': df[numeric_columns].min(),
        'max': df[numeric_columns].max(),
        'skewness': df[numeric_columns].apply(lambda x: stats.skew(x.dropna())),
        'kurtosis': df[numeric_columns].apply(lambda x: stats.kurtosis(x.dropna()))
    }
    stats_df = pd.DataFrame(stats_dict)
    return stats_df