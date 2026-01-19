
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    return dataframe[(dataframe[column] >= lower_bound) & 
                     (dataframe[column] <= upper_bound)]

def remove_outliers_zscore(dataframe, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    z_scores = np.abs(stats.zscore(dataframe[column]))
    return dataframe[z_scores < threshold]

def normalize_minmax(dataframe, column):
    """
    Normalize data using Min-Max scaling
    """
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()
    
    if max_val == min_val:
        return dataframe[column].apply(lambda x: 0.5)
    
    return (dataframe[column] - min_val) / (max_val - min_val)

def normalize_zscore(dataframe, column):
    """
    Normalize data using Z-score standardization
    """
    mean_val = dataframe[column].mean()
    std_val = dataframe[column].std()
    
    if std_val == 0:
        return dataframe[column].apply(lambda x: 0)
    
    return (dataframe[column] - mean_val) / std_val

def clean_dataset(dataframe, numeric_columns, method='iqr', normalize=False):
    """
    Main cleaning function for numeric columns
    """
    cleaned_df = dataframe.copy()
    
    for column in numeric_columns:
        if column not in cleaned_df.columns:
            continue
            
        if method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
        elif method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, column)
        
        if normalize:
            norm_method = 'minmax' if method == 'iqr' else 'zscore'
            if norm_method == 'minmax':
                cleaned_df[column] = normalize_minmax(cleaned_df, column)
            else:
                cleaned_df[column] = normalize_zscore(cleaned_df, column)
    
    return cleaned_df

def validate_dataframe(dataframe, required_columns):
    """
    Validate dataframe structure and content
    """
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if dataframe.empty:
        raise ValueError("DataFrame is empty")
    
    return True

def get_summary_statistics(dataframe):
    """
    Generate summary statistics for numeric columns
    """
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return pd.DataFrame()
    
    summary = dataframe[numeric_cols].describe()
    summary.loc['missing'] = dataframe[numeric_cols].isnull().sum()
    summary.loc['zeros'] = (dataframe[numeric_cols] == 0).sum()
    
    return summary