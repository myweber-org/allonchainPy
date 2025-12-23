
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def clean_dataset(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
        print(f"Original dataset shape: {df.shape}")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            original_count = len(df)
            df = remove_outliers_iqr(df, col)
            removed_count = original_count - len(df)
            if removed_count > 0:
                print(f"Removed {removed_count} outliers from column: {col}")
        
        df.to_csv(output_file, index=False)
        print(f"Cleaned dataset saved to: {output_file}")
        print(f"Final dataset shape: {df.shape}")
        
        return True
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return False
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return False

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    clean_dataset(input_file, output_file)import numpy as np
import pandas as pd
from scipy import stats

def normalize_data(data, method='zscore'):
    if method == 'zscore':
        return (data - np.mean(data)) / np.std(data)
    elif method == 'minmax':
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    else:
        raise ValueError("Method must be 'zscore' or 'minmax'")

def remove_outliers_iqr(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return data[(data >= lower_bound) & (data <= upper_bound)]

def remove_outliers_zscore(data, threshold=3):
    z_scores = np.abs(stats.zscore(data))
    return data[z_scores < threshold]

def clean_dataset(df, column, outlier_method='iqr', normalize=True):
    cleaned_data = df[column].copy()
    
    if outlier_method == 'iqr':
        cleaned_data = remove_outliers_iqr(cleaned_data)
    elif outlier_method == 'zscore':
        cleaned_data = remove_outliers_zscore(cleaned_data)
    
    if normalize:
        cleaned_data = normalize_data(cleaned_data)
    
    return cleaned_data

def process_dataframe(df, numeric_columns=None):
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    processed_df = df.copy()
    for col in numeric_columns:
        if col in df.columns:
            processed_df[col] = clean_dataset(df, col)
    
    return processed_df

def validate_data(data):
    if isinstance(data, pd.DataFrame):
        return not data.isnull().any().any()
    elif isinstance(data, np.ndarray):
        return not np.isnan(data).any()
    else:
        return not pd.isnull(data)