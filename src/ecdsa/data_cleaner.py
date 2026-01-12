import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    """
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(df, numeric_columns, method='iqr', normalization=None):
    """
    Main cleaning function that handles outliers and optional normalization.
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        
        if normalization == 'minmax':
            cleaned_df = normalize_minmax(cleaned_df, col)
        elif normalization == 'zscore':
            cleaned_df = normalize_zscore(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, numeric_columns):
    """
    Validate data by checking for missing values and infinite values.
    """
    validation_report = {}
    
    for col in numeric_columns:
        validation_report[col] = {
            'missing_values': df[col].isnull().sum(),
            'infinite_values': np.isinf(df[col]).sum(),
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max()
        }
    
    return pd.DataFrame(validation_report).T

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'feature3': np.random.uniform(0, 200, 1000)
    })
    
    # Add some outliers
    sample_data.loc[1000] = [500, 1000, 300]
    sample_data.loc[1001] = [-100, -50, -10]
    
    print("Original data shape:", sample_data.shape)
    print("\nOriginal statistics:")
    print(validate_data(sample_data, ['feature1', 'feature2', 'feature3']))
    
    # Clean data
    cleaned_data = clean_dataset(
        sample_data, 
        ['feature1', 'feature2', 'feature3'], 
        method='iqr', 
        normalization='zscore'
    )
    
    print("\nCleaned data shape:", cleaned_data.shape)
    print("\nCleaned statistics:")
    print(validate_data(cleaned_data, ['feature1', 'feature2', 'feature3']))