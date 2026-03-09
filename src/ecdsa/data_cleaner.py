
import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_strategy='mean', drop_threshold=0.5):
    """
    Load and clean CSV data by handling missing values.
    
    Parameters:
    filepath (str): Path to the CSV file
    fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'zero')
    drop_threshold (float): Drop columns with missing values above this threshold (0.0 to 1.0)
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame
    """
    
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except Exception as e:
        raise Exception(f"Error reading file: {str(e)}")
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    missing_percentage = df.isnull().sum() / len(df)
    columns_to_drop = missing_percentage[missing_percentage > drop_threshold].index
    df = df.drop(columns=columns_to_drop)
    
    if len(columns_to_drop) > 0:
        print(f"Dropped columns with >{drop_threshold*100}% missing values: {list(columns_to_drop)}")
    
    for column in df.columns:
        if df[column].dtype in ['float64', 'int64']:
            if fill_strategy == 'mean':
                fill_value = df[column].mean()
            elif fill_strategy == 'median':
                fill_value = df[column].median()
            elif fill_strategy == 'mode':
                fill_value = df[column].mode()[0] if not df[column].mode().empty else 0
            elif fill_strategy == 'zero':
                fill_value = 0
            else:
                raise ValueError(f"Unknown fill strategy: {fill_strategy}")
            
            df[column] = df[column].fillna(fill_value)
        else:
            df[column] = df[column].fillna(df[column].mode()[0] if not df[column].mode().empty else 'Unknown')
    
    print(f"Cleaned data shape: {df.shape}")
    print(f"Missing values after cleaning: {df.isnull().sum().sum()}")
    
    return df

def detect_outliers_iqr(df, column):
    """
    Detect outliers using IQR method for a specific column.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    column (str): Column name to check for outliers
    
    Returns:
    pandas.Series: Boolean series indicating outliers
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not np.issubdtype(df[column].dtype, np.number):
        raise ValueError(f"Column '{column}' must be numeric for outlier detection")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    return outliers

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to CSV.
    
    Parameters:
    df (pandas.DataFrame): Cleaned DataFrame
    output_path (str): Path to save the cleaned CSV file
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, np.nan, np.nan, 4, 5],
        'C': [10, 20, 30, 40, 50],
        'D': ['a', 'b', np.nan, 'd', 'e']
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', fill_strategy='mean', drop_threshold=0.6)
    print("\nSample cleaned data:")
    print(cleaned_df)
    
    import os
    if os.path.exists('sample_data.csv'):
        os.remove('sample_data.csv')import numpy as np
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
    """Normalize data using Min-Max scaling."""
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def normalize_standard(data, column):
    """Normalize data using Standardization (Z-score normalization)."""
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(df, numeric_columns, method='iqr', normalize=False):
    """Main function to clean dataset by removing outliers and optionally normalizing."""
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        
        if normalize:
            cleaned_df = normalize_minmax(cleaned_df, col)
    
    return cleaned_df.reset_index(drop=True)

def validate_data(df, required_columns):
    """Validate that required columns exist and have no null values."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        print(f"Warning: Found null values in columns: {null_counts[null_counts > 0].to_dict()}")
    
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    print("Original data shape:", sample_data.shape)
    cleaned_data = clean_dataset(sample_data, ['feature_a', 'feature_b'], method='iqr', normalize=True)
    print("Cleaned data shape:", cleaned_data.shape)
    print("Cleaned data columns:", cleaned_data.columns.tolist())