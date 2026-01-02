import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling
    """
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values with different strategies
    """
    data_clean = data.copy()
    
    for column in data_clean.columns:
        if data_clean[column].isnull().any():
            if strategy == 'mean':
                fill_value = data_clean[column].mean()
            elif strategy == 'median':
                fill_value = data_clean[column].median()
            elif strategy == 'mode':
                fill_value = data_clean[column].mode()[0]
            elif strategy == 'drop':
                data_clean = data_clean.dropna(subset=[column])
                continue
            else:
                fill_value = 0
            
            data_clean[column] = data_clean[column].fillna(fill_value)
    
    return data_clean

def detect_anomalies_grubbs(data, column, alpha=0.05):
    """
    Detect anomalies using Grubbs' test
    """
    values = data[column].dropna().values
    n = len(values)
    
    if n < 3:
        return []
    
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    # Calculate Grubbs' statistic
    g_calculated = np.max(np.abs(values - mean_val)) / std_val
    
    # Critical value
    t_dist = stats.t.ppf(1 - alpha/(2*n), n-2)
    g_critical = ((n-1) * t_dist) / np.sqrt(n*(n-2 + t_dist**2))
    
    anomalies = []
    if g_calculated > g_critical:
        # Find the outlier
        outlier_idx = np.argmax(np.abs(values - mean_val))
        anomalies.append(values[outlier_idx])
    
    return anomalies

def create_cleaning_pipeline(data, config):
    """
    Apply multiple cleaning steps based on configuration
    """
    cleaned_data = data.copy()
    
    for step in config.get('steps', []):
        if step['type'] == 'remove_outliers':
            cleaned_data = remove_outliers_iqr(
                cleaned_data, 
                step['column'], 
                step.get('threshold', 1.5)
            )
        elif step['type'] == 'normalize':
            cleaned_data[step['column']] = normalize_minmax(
                cleaned_data, 
                step['column']
            )
        elif step['type'] == 'standardize':
            cleaned_data[step['column']] = standardize_zscore(
                cleaned_data, 
                step['column']
            )
        elif step['type'] == 'handle_missing':
            cleaned_data = handle_missing_values(
                cleaned_data, 
                step.get('strategy', 'mean')
            )
    
    return cleaned_data

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 100),
        'feature2': np.random.exponential(50, 100),
        'feature3': np.random.uniform(0, 1, 100)
    })
    
    # Add some outliers
    sample_data.loc[0, 'feature1'] = 300
    sample_data.loc[1, 'feature2'] = 500
    
    # Add missing values
    sample_data.loc[5:10, 'feature3'] = np.nan
    
    print("Original data shape:", sample_data.shape)
    print("Missing values:", sample_data.isnull().sum().sum())
    
    # Define cleaning pipeline
    cleaning_config = {
        'steps': [
            {'type': 'handle_missing', 'strategy': 'mean'},
            {'type': 'remove_outliers', 'column': 'feature1', 'threshold': 1.5},
            {'type': 'remove_outliers', 'column': 'feature2', 'threshold': 1.5},
            {'type': 'standardize', 'column': 'feature1'},
            {'type': 'normalize', 'column': 'feature2'}
        ]
    }
    
    # Apply cleaning pipeline
    cleaned = create_cleaning_pipeline(sample_data, cleaning_config)
    
    print("\nCleaned data shape:", cleaned.shape)
    print("Missing values after cleaning:", cleaned.isnull().sum().sum())
    print("\nFeature1 stats - Mean:", cleaned['feature1'].mean(), 
          "Std:", cleaned['feature1'].std())
    print("Feature2 stats - Min:", cleaned['feature2'].min(), 
          "Max:", cleaned['feature2'].max())