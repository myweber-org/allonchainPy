import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, columns):
    """
    Normalize specified columns using Min-Max scaling.
    """
    normalized_data = data.copy()
    for col in columns:
        min_val = normalized_data[col].min()
        max_val = normalized_data[col].max()
        if max_val != min_val:
            normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
        else:
            normalized_data[col] = 0
    return normalized_data

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy.
    """
    processed_data = data.copy()
    if columns is None:
        columns = processed_data.columns
    
    for col in columns:
        if processed_data[col].isnull().any():
            if strategy == 'mean':
                fill_value = processed_data[col].mean()
            elif strategy == 'median':
                fill_value = processed_data[col].median()
            elif strategy == 'mode':
                fill_value = processed_data[col].mode()[0]
            elif strategy == 'drop':
                processed_data = processed_data.dropna(subset=[col])
                continue
            else:
                fill_value = 0
            
            processed_data[col] = processed_data[col].fillna(fill_value)
    
    return processed_data

def clean_dataset(data, config):
    """
    Main function to clean dataset based on configuration.
    """
    cleaned_data = data.copy()
    
    if 'outlier_removal' in config:
        for col in config['outlier_removal'].get('columns', []):
            cleaned_data = remove_outliers_iqr(
                cleaned_data, 
                col, 
                factor=config['outlier_removal'].get('factor', 1.5)
            )
    
    if 'normalization' in config:
        cleaned_data = normalize_minmax(
            cleaned_data, 
            config['normalization'].get('columns', [])
        )
    
    if 'missing_values' in config:
        cleaned_data = handle_missing_values(
            cleaned_data,
            strategy=config['missing_values'].get('strategy', 'mean'),
            columns=config['missing_values'].get('columns')
        )
    
    return cleaned_data

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.uniform(0, 1, 1000),
        'C': np.random.choice([1, 2, 3, None], 1000, p=[0.3, 0.3, 0.3, 0.1])
    })
    
    config = {
        'outlier_removal': {
            'columns': ['A'],
            'factor': 1.5
        },
        'normalization': {
            'columns': ['B']
        },
        'missing_values': {
            'strategy': 'mean',
            'columns': ['C']
        }
    }
    
    cleaned = clean_dataset(sample_data, config)
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned.shape}")
    print(f"Missing values after cleaning: {cleaned.isnull().sum().sum()}")