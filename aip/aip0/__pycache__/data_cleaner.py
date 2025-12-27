
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
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
    column (str): Column name
    
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

def validate_numeric_data(df, columns):
    """
    Validate that specified columns contain only numeric data.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to validate
    
    Returns:
    dict: Dictionary with validation results for each column
    """
    validation_results = {}
    
    for col in columns:
        if col not in df.columns:
            validation_results[col] = {'valid': False, 'error': 'Column not found'}
            continue
        
        non_numeric = pd.to_numeric(df[col], errors='coerce').isna().sum()
        total_rows = len(df)
        
        validation_results[col] = {
            'valid': non_numeric == 0,
            'non_numeric_count': non_numeric,
            'non_numeric_percentage': (non_numeric / total_rows * 100) if total_rows > 0 else 0,
            'total_rows': total_rows
        }
    
    return validation_results

def example_usage():
    """
    Example demonstrating the usage of data cleaning functions.
    """
    np.random.seed(42)
    
    data = {
        'id': range(1, 101),
        'value': np.random.normal(100, 15, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    
    cleaned_df = remove_outliers_iqr(df, 'value')
    print("Cleaned DataFrame shape:", cleaned_df.shape)
    
    stats = calculate_summary_stats(cleaned_df, 'value')
    print("\nSummary Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")
    
    validation = validate_numeric_data(df, ['value', 'id', 'category'])
    print("\nData Validation Results:")
    for col, result in validation.items():
        print(f"{col}: {result}")

if __name__ == "__main__":
    example_usage()