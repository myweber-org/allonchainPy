import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    """
    df_clean = df.copy()
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean.reset_index(drop=True)

def remove_outliers_zscore(df, columns, threshold=3):
    """
    Remove outliers using Z-score method.
    """
    df_clean = df.copy()
    for col in columns:
        if col in df.columns:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            df_clean = df_clean[(z_scores < threshold) | (df[col].isna())]
    return df_clean.reset_index(drop=True)

def normalize_minmax(df, columns):
    """
    Normalize data using Min-Max scaling.
    """
    df_norm = df.copy()
    for col in columns:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df_norm[col] = (df[col] - min_val) / (max_val - min_val)
    return df_norm

def normalize_zscore(df, columns):
    """
    Normalize data using Z-score standardization.
    """
    df_norm = df.copy()
    for col in columns:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                df_norm[col] = (df[col] - mean_val) / std_val
    return df_norm

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values with specified strategy.
    """
    df_filled = df.copy()
    if columns is None:
        columns = df.columns
    
    for col in columns:
        if col in df.columns:
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
            elif strategy == 'zero':
                fill_value = 0
            else:
                fill_value = df[col].mean()
            
            df_filled[col] = df[col].fillna(fill_value)
    
    return df_filled

def get_data_summary(df):
    """
    Generate comprehensive data summary statistics.
    """
    summary = pd.DataFrame({
        'dtype': df.dtypes,
        'non_null_count': df.count(),
        'null_count': df.isnull().sum(),
        'null_percentage': (df.isnull().sum() / len(df)) * 100,
        'unique_count': df.nunique(),
        'mean': df.select_dtypes(include=[np.number]).mean(),
        'std': df.select_dtypes(include=[np.number]).std(),
        'min': df.select_dtypes(include=[np.number]).min(),
        '25%': df.select_dtypes(include=[np.number]).quantile(0.25),
        '50%': df.select_dtypes(include=[np.number]).quantile(0.50),
        '75%': df.select_dtypes(include=[np.number]).quantile(0.75),
        'max': df.select_dtypes(include=[np.number]).max()
    })
    
    return summary
import pandas as pd
import re

def clean_dataframe(df, text_columns=None):
    """
    Remove duplicate rows and standardize text in specified columns.
    """
    df_clean = df.copy()
    
    # Remove duplicates
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    removed_duplicates = initial_rows - len(df_clean)
    
    # Standardize text in specified columns
    if text_columns:
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].apply(_standardize_text)
    
    return df_clean, removed_duplicates

def _standardize_text(text):
    """
    Helper function to standardize text: lowercase, remove extra spaces.
    """
    if pd.isna(text):
        return text
    text = str(text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()