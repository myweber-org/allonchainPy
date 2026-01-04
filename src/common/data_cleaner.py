import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    """Remove outliers using IQR method."""
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
    """Normalize data using min-max scaling."""
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def normalize_zscore(data, column):
    """Normalize data using Z-score standardization."""
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def handle_missing_values(data, strategy='mean'):
    """Handle missing values with specified strategy."""
    if strategy == 'mean':
        return data.fillna(data.mean())
    elif strategy == 'median':
        return data.fillna(data.median())
    elif strategy == 'mode':
        return data.fillna(data.mode().iloc[0])
    elif strategy == 'drop':
        return data.dropna()
    else:
        raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'mode', or 'drop'.")

def clean_dataset(data, numeric_columns, outlier_method='iqr', normalize_method=None, missing_strategy='mean'):
    """Main function to clean dataset with multiple options."""
    cleaned_data = data.copy()
    
    for col in numeric_columns:
        if col in cleaned_data.columns:
            cleaned_data = handle_missing_values(cleaned_data[[col]], strategy=missing_strategy)
            
            if outlier_method == 'iqr':
                cleaned_data = remove_outliers_iqr(cleaned_data, col)
            elif outlier_method == 'zscore':
                cleaned_data = remove_outliers_zscore(cleaned_data, col)
            
            if normalize_method == 'minmax':
                cleaned_data = normalize_minmax(cleaned_data, col)
            elif normalize_method == 'zscore':
                cleaned_data = normalize_zscore(cleaned_data, col)
    
    return cleaned_data

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'age': [25, 30, 35, 200, 28, 32, 150, 29, np.nan, 31],
        'salary': [50000, 55000, 60000, 1000000, 52000, 58000, 900000, 53000, 54000, np.nan],
        'score': [85, 90, 88, 95, 82, 91, 99, 87, 89, 92]
    })
    
    print("Original Data:")
    print(sample_data)
    print("\nCleaned Data (IQR outlier removal, min-max normalization):")
    cleaned = clean_dataset(
        sample_data, 
        numeric_columns=['age', 'salary', 'score'],
        outlier_method='iqr',
        normalize_method='minmax',
        missing_strategy='mean'
    )
    print(cleaned)