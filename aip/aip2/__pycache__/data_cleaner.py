
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, multiplier=1.5):
    """
    Remove outliers using IQR method
    """
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]

def remove_outliers_zscore(dataframe, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    z_scores = np.abs(stats.zscore(dataframe[column]))
    return dataframe[z_scores < threshold]

def normalize_minmax(dataframe, column):
    """
    Normalize column using Min-Max scaling
    """
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()
    
    if max_val == min_val:
        return dataframe[column].apply(lambda x: 0.5)
    
    return (dataframe[column] - min_val) / (max_val - min_val)

def normalize_zscore(dataframe, column):
    """
    Normalize column using Z-score standardization
    """
    mean_val = dataframe[column].mean()
    std_val = dataframe[column].std()
    
    if std_val == 0:
        return dataframe[column].apply(lambda x: 0)
    
    return (dataframe[column] - mean_val) / std_val

def clean_dataset(dataframe, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    """
    Main cleaning function for numeric columns
    """
    cleaned_df = dataframe.copy()
    
    for column in numeric_columns:
        if column not in cleaned_df.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, column)
        
        if normalize_method == 'minmax':
            cleaned_df[column] = normalize_minmax(cleaned_df, column)
        elif normalize_method == 'zscore':
            cleaned_df[column] = normalize_zscore(cleaned_df, column)
    
    return cleaned_df.reset_index(drop=True)

def get_summary_statistics(dataframe, numeric_columns):
    """
    Generate summary statistics for numeric columns
    """
    summary = {}
    
    for column in numeric_columns:
        if column in dataframe.columns:
            col_data = dataframe[column]
            summary[column] = {
                'mean': col_data.mean(),
                'median': col_data.median(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max(),
                'count': col_data.count(),
                'missing': col_data.isnull().sum()
            }
    
    return pd.DataFrame(summary).T

def detect_skewness(dataframe, column, threshold=0.5):
    """
    Detect skewness in column data
    """
    skewness = dataframe[column].skew()
    is_skewed = abs(skewness) > threshold
    
    return {
        'skewness': skewness,
        'is_skewed': is_skewed,
        'direction': 'right' if skewness > 0 else 'left' if skewness < 0 else 'none'
    }
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

def normalize_minmax(df, columns):
    normalized_df = df.copy()
    for col in columns:
        if col in normalized_df.columns:
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            if max_val != min_val:
                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
    return normalized_df

def handle_missing_values(df, strategy='mean'):
    handled_df = df.copy()
    numeric_cols = handled_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if handled_df[col].isnull().any():
            if strategy == 'mean':
                fill_value = handled_df[col].mean()
            elif strategy == 'median':
                fill_value = handled_df[col].median()
            elif strategy == 'mode':
                fill_value = handled_df[col].mode()[0]
            else:
                fill_value = 0
            handled_df[col] = handled_df[col].fillna(fill_value)
    
    return handled_df

def clean_dataset(df, numeric_columns=None, outlier_method='iqr', normalize=True, missing_strategy='mean'):
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_cleaned = df.copy()
    
    df_cleaned = handle_missing_values(df_cleaned, strategy=missing_strategy)
    
    if outlier_method == 'iqr':
        df_cleaned = remove_outliers_iqr(df_cleaned, numeric_columns)
    elif outlier_method == 'zscore':
        z_scores = np.abs(stats.zscore(df_cleaned[numeric_columns]))
        df_cleaned = df_cleaned[(z_scores < 3).all(axis=1)]
    
    if normalize:
        df_cleaned = normalize_minmax(df_cleaned, numeric_columns)
    
    return df_cleaned