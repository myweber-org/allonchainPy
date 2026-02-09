import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    """Remove outliers using the IQR method."""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    """Remove outliers using Z-score method."""
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    """Normalize data to [0, 1] range."""
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    return (data[column] - min_val) / (max_val - min_val)

def normalize_zscore(data, column):
    """Normalize data using Z-score normalization."""
    mean_val = data[column].mean()
    std_val = data[column].std()
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    return (data[column] - mean_val) / std_val

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    """Apply outlier removal and normalization to specified columns."""
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col not in cleaned_df.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        
        if normalize_method == 'minmax':
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df[col] = normalize_zscore(cleaned_df, col)
    
    return cleaned_df

def get_summary_statistics(df):
    """Calculate summary statistics for numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary = df[numeric_cols].describe()
    summary.loc['skewness'] = df[numeric_cols].skew()
    summary.loc['kurtosis'] = df[numeric_cols].kurtosis()
    return summaryimport re
import pandas as pd

def normalize_string(text):
    if not isinstance(text, str):
        return text
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    return text

def clean_numeric(value):
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = re.sub(r'[^\d.-]', '', value)
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None

def remove_duplicates(df, subset=None):
    if subset is None:
        subset = df.columns.tolist()
    return df.drop_duplicates(subset=subset, keep='first')

def validate_email(email):
    if not isinstance(email, str):
        return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def process_dataframe(df, string_columns=None, numeric_columns=None):
    df_clean = df.copy()
    
    if string_columns:
        for col in string_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].apply(normalize_string)
    
    if numeric_columns:
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].apply(clean_numeric)
    
    return df_clean