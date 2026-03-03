
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
    df = pd.read_csv(input_file)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for column in numeric_columns:
        df = remove_outliers_iqr(df, column)
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    print(f"Original rows: {len(pd.read_csv(input_file))}")
    print(f"Cleaned rows: {len(df)}")

if __name__ == "__main__":
    clean_dataset("raw_data.csv", "cleaned_data.csv")
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(filepath):
    """Load dataset from CSV file"""
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, columns):
    """Remove outliers using IQR method"""
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def remove_outliers_zscore(df, columns, threshold=3):
    """Remove outliers using Z-score method"""
    df_clean = df.copy()
    for col in columns:
        z_scores = np.abs(stats.zscore(df_clean[col]))
        df_clean = df_clean[z_scores < threshold]
    return df_clean

def normalize_minmax(df, columns):
    """Normalize data using Min-Max scaling"""
    df_normalized = df.copy()
    for col in columns:
        min_val = df_normalized[col].min()
        max_val = df_normalized[col].max()
        df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
    return df_normalized

def normalize_zscore(df, columns):
    """Normalize data using Z-score standardization"""
    df_normalized = df.copy()
    for col in columns:
        mean_val = df_normalized[col].mean()
        std_val = df_normalized[col].std()
        df_normalized[col] = (df_normalized[col] - mean_val) / std_val
    return df_normalized

def handle_missing_values(df, strategy='mean'):
    """Handle missing values with specified strategy"""
    df_filled = df.copy()
    numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        for col in numeric_cols:
            df_filled[col].fillna(df_filled[col].mean(), inplace=True)
    elif strategy == 'median':
        for col in numeric_cols:
            df_filled[col].fillna(df_filled[col].median(), inplace=True)
    elif strategy == 'mode':
        for col in numeric_cols:
            df_filled[col].fillna(df_filled[col].mode()[0], inplace=True)
    elif strategy == 'drop':
        df_filled.dropna(inplace=True)
    
    return df_filled

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax', missing_strategy='mean'):
    """Complete data cleaning pipeline"""
    print(f"Original dataset shape: {df.shape}")
    
    df_cleaned = handle_missing_values(df, strategy=missing_strategy)
    print(f"After handling missing values: {df_cleaned.shape}")
    
    if outlier_method == 'iqr':
        df_cleaned = remove_outliers_iqr(df_cleaned, numeric_columns)
    elif outlier_method == 'zscore':
        df_cleaned = remove_outliers_zscore(df_cleaned, numeric_columns)
    print(f"After outlier removal: {df_cleaned.shape}")
    
    if normalize_method == 'minmax':
        df_cleaned = normalize_minmax(df_cleaned, numeric_columns)
    elif normalize_method == 'zscore':
        df_cleaned = normalize_zscore(df_cleaned, numeric_columns)
    
    return df_cleaned

def save_cleaned_data(df, output_path):
    """Save cleaned dataset to CSV"""
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    numeric_cols = ['age', 'income', 'score']
    
    try:
        raw_data = load_dataset(input_file)
        cleaned_data = clean_dataset(
            raw_data, 
            numeric_columns=numeric_cols,
            outlier_method='iqr',
            normalize_method='zscore',
            missing_strategy='median'
        )
        save_cleaned_data(cleaned_data, output_file)
        print("Data cleaning completed successfully!")
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        if max_val != min_val:
            df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val != 0:
            df_copy[column] = (df_copy[column] - mean_val) / std_val
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df_copy

if __name__ == "__main__":
    sample_data = {
        'values': [10, 12, 12, 13, 12, 11, 100, 14, 12, 10, 9, 8, 15, 200]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("DataFrame after removing outliers:")
    print(cleaned_df)
    print()
    
    stats = calculate_summary_stats(df, 'values')
    print("Summary statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print()
    
    normalized_df = normalize_column(df, 'values', method='minmax')
    print("Normalized DataFrame (minmax):")
    print(normalized_df)