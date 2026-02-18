import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    filtered_data = data[(z_scores < threshold) | (data[column].isna())]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_data(data, column):
    """
    Standardize data using Z-score normalization
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize=False):
    """
    Main cleaning function for datasets
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col not in cleaned_df.columns:
            continue
            
        # Handle missing values
        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        
        # Remove outliers
        if outlier_method == 'iqr':
            outliers, lower, upper = detect_outliers_iqr(cleaned_df, col)
            cleaned_df = cleaned_df[(cleaned_df[col] >= lower) & (cleaned_df[col] <= upper)]
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        
        # Normalize if requested
        if normalize:
            cleaned_df[f'{col}_normalized'] = normalize_minmax(cleaned_df, col)
            cleaned_df[f'{col}_standardized'] = standardize_data(cleaned_df, col)
    
    return cleaned_df

def validate_data_quality(df, numeric_columns):
    """
    Validate data quality metrics
    """
    quality_report = {}
    
    for col in numeric_columns:
        if col not in df.columns:
            continue
            
        quality_report[col] = {
            'missing_count': df[col].isna().sum(),
            'missing_percentage': (df[col].isna().sum() / len(df)) * 100,
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'skewness': df[col].skew(),
            'kurtosis': df[col].kurtosis()
        }
    
    return pd.DataFrame(quality_report).T

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 1, 1000)
    })
    
    # Add some outliers
    sample_data.loc[10, 'feature_a'] = 500
    sample_data.loc[20, 'feature_b'] = 1000
    
    # Clean the data
    numeric_cols = ['feature_a', 'feature_b', 'feature_c']
    cleaned_data = clean_dataset(sample_data, numeric_cols, outlier_method='iqr', normalize=True)
    
    # Validate quality
    quality_report = validate_data_quality(cleaned_data, numeric_cols)
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned_data.shape}")
    print("\nQuality Report:")
    print(quality_report)