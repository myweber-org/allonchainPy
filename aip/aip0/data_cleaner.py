
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
        'count': len(df[column]),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    data = {
        'values': np.concatenate([
            np.random.normal(100, 15, 90),
            np.random.normal(300, 50, 10)
        ])
    }
    
    df = pd.DataFrame(data)
    print(f"Original data shape: {df.shape}")
    print(f"Original statistics: {calculate_summary_stats(df, 'values')}")
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print(f"\nCleaned data shape: {cleaned_df.shape}")
    print(f"Cleaned statistics: {calculate_summary_stats(cleaned_df, 'values')}")

if __name__ == "__main__":
    example_usage()
import pandas as pd

def clean_dataset(df, missing_strategy='drop', duplicate_strategy='drop_first'):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    missing_strategy (str): Strategy for missing values. 
                            'drop': Drop rows with any missing values.
                            'fill_mean': Fill numeric columns with mean.
                            'fill_median': Fill numeric columns with median.
    duplicate_strategy (str): Strategy for duplicates.
                              'drop_first': Keep first occurrence, drop duplicates.
                              'drop_last': Keep last occurrence, drop duplicates.
                              'keep_none': Drop all duplicates.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif missing_strategy in ['fill_mean', 'fill_median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if missing_strategy == 'fill_mean':
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            elif missing_strategy == 'fill_median':
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    
    # Handle duplicates
    if duplicate_strategy == 'drop_first':
        cleaned_df = cleaned_df.drop_duplicates(keep='first')
    elif duplicate_strategy == 'drop_last':
        cleaned_df = cleaned_df.drop_duplicates(keep='last')
    elif duplicate_strategy == 'keep_none':
        cleaned_df = cleaned_df.drop_duplicates(keep=False)
    
    return cleaned_df

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (is_valid, message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Dataset is valid"

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10.5, None, 15.2, 20.1, 10.5, 10.5],
        'category': ['A', 'B', 'A', None, 'B', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nShape:", df.shape)
    
    # Clean the data
    cleaned = clean_dataset(df, missing_strategy='fill_mean', duplicate_strategy='drop_first')
    print("\nCleaned DataFrame:")
    print(cleaned)
    print("\nShape:", cleaned.shape)
    
    # Validate
    is_valid, message = validate_dataset(cleaned, required_columns=['id', 'value'], min_rows=2)
    print(f"\nValidation: {is_valid} - {message}")