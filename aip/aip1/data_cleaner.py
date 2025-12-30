
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(dataframe, columns):
    cleaned_df = dataframe.copy()
    for col in columns:
        if col in cleaned_df.columns:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    return cleaned_df

def normalize_minmax(dataframe, columns):
    normalized_df = dataframe.copy()
    for col in columns:
        if col in normalized_df.columns:
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            if max_val != min_val:
                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
    return normalized_df

def handle_missing_values(dataframe, strategy='mean'):
    processed_df = dataframe.copy()
    for col in processed_df.columns:
        if processed_df[col].isnull().any():
            if strategy == 'mean':
                fill_value = processed_df[col].mean()
            elif strategy == 'median':
                fill_value = processed_df[col].median()
            elif strategy == 'mode':
                fill_value = processed_df[col].mode()[0]
            else:
                fill_value = 0
            processed_df[col].fillna(fill_value, inplace=True)
    return processed_df

def calculate_statistics(dataframe):
    stats_dict = {}
    for col in dataframe.select_dtypes(include=[np.number]).columns:
        stats_dict[col] = {
            'mean': dataframe[col].mean(),
            'std': dataframe[col].std(),
            'median': dataframe[col].median(),
            'min': dataframe[col].min(),
            'max': dataframe[col].max()
        }
    return stats_dict

def process_dataset(file_path, outlier_columns=None, normalize_columns=None):
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded dataset with shape: {df.shape}")
        
        if outlier_columns:
            df = remove_outliers_iqr(df, outlier_columns)
            print(f"After outlier removal: {df.shape}")
        
        df = handle_missing_values(df)
        
        if normalize_columns:
            df = normalize_minmax(df, normalize_columns)
        
        stats = calculate_statistics(df)
        return df, stats
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return None, None