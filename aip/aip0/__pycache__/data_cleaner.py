
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None, missing_strategy='drop'):
    """
    Clean CSV data by handling missing values and standardizing columns.
    
    Parameters:
    input_path (str): Path to input CSV file
    output_path (str, optional): Path for cleaned output CSV. If None, returns DataFrame
    missing_strategy (str): Strategy for handling missing values: 'drop', 'mean', 'median', or 'zero'
    
    Returns:
    pd.DataFrame or None: Cleaned DataFrame if output_path is None, else None
    """
    
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    
    if missing_strategy == 'drop':
        df_cleaned = df.dropna()
    elif missing_strategy == 'mean' and len(numeric_cols) > 0:
        df_cleaned = df.copy()
        for col in numeric_cols:
            df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
    elif missing_strategy == 'median' and len(numeric_cols) > 0:
        df_cleaned = df.copy()
        for col in numeric_cols:
            df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
    elif missing_strategy == 'zero' and len(numeric_cols) > 0:
        df_cleaned = df.copy()
        for col in numeric_cols:
            df_cleaned[col].fillna(0, inplace=True)
    else:
        df_cleaned = df.copy()
    
    for col in categorical_cols:
        df_cleaned[col].fillna('Unknown', inplace=True)
    
    cleaned_shape = df_cleaned.shape
    rows_removed = original_shape[0] - cleaned_shape[0]
    
    print(f"Cleaned data shape: {cleaned_shape}")
    print(f"Rows removed due to missing values: {rows_removed}")
    
    if output_path:
        df_cleaned.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        return None
    else:
        return df_cleaned

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list, optional): List of required column names
    
    Returns:
    dict: Validation results
    """
    
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append("DataFrame is empty")
    
    validation_results['stats']['total_rows'] = len(df)
    validation_results['stats']['total_columns'] = len(df.columns)
    validation_results['stats']['numeric_columns'] = len(df.select_dtypes(include=[np.number]).columns)
    validation_results['stats']['categorical_columns'] = len(df.select_dtypes(exclude=[np.number]).columns)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            validation_results['warnings'].append(f"Column '{col}' contains missing values")
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5],
        'value': [10.5, np.nan, 15.2, np.nan, 20.1],
        'category': ['A', 'B', 'A', np.nan, 'C'],
        'score': [85, 92, 78, 88, np.nan]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', missing_strategy='mean')
    
    validation = validate_dataframe(cleaned_df, required_columns=['id', 'value', 'category'])
    print("\nValidation Results:")
    print(f"Is Valid: {validation['is_valid']}")
    print(f"Errors: {validation['errors']}")
    print(f"Warnings: {validation['warnings']}")
    print(f"Stats: {validation['stats']}")
    
    import os
    if os.path.exists('test_data.csv'):
        os.remove('test_data.csv')
import pandas as pd
import numpy as np

def clean_dataset(df, strategy='mean', outlier_threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Strategy for missing values: 'mean', 'median', 'mode', or 'drop'
    outlier_threshold (float): Number of standard deviations for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    for column in cleaned_df.columns:
        if cleaned_df[column].dtype in ['float64', 'int64']:
            if strategy == 'mean':
                cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
            elif strategy == 'median':
                cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
            elif strategy == 'mode':
                cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
            elif strategy == 'drop':
                cleaned_df.dropna(subset=[column], inplace=True)
    
    # Handle outliers using z-score method
    numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        z_scores = np.abs((cleaned_df[column] - cleaned_df[column].mean()) / cleaned_df[column].std())
        cleaned_df = cleaned_df[z_scores < outlier_threshold]
    
    # Reset index after cleaning
    cleaned_df.reset_index(drop=True, inplace=True)
    
    return cleaned_df

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    subset (list): Columns to consider for duplicates
    keep (str): Which duplicates to keep: 'first', 'last', or False
    
    Returns:
    pd.DataFrame: DataFrame without duplicates
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def normalize_columns(df, columns=None, method='minmax'):
    """
    Normalize specified columns in DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): Columns to normalize, None for all numeric columns
    method (str): Normalization method: 'minmax' or 'zscore'
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    normalized_df = df.copy()
    
    if columns is None:
        columns = normalized_df.select_dtypes(include=[np.number]).columns
    
    for column in columns:
        if column in normalized_df.columns and normalized_df[column].dtype in ['float64', 'int64']:
            if method == 'minmax':
                col_min = normalized_df[column].min()
                col_max = normalized_df[column].max()
                if col_max != col_min:
                    normalized_df[column] = (normalized_df[column] - col_min) / (col_max - col_min)
            elif method == 'zscore':
                col_mean = normalized_df[column].mean()
                col_std = normalized_df[column].std()
                if col_std != 0:
                    normalized_df[column] = (normalized_df[column] - col_mean) / col_std
    
    return normalized_df

# Example usage
if __name__ == "__main__":
    # Create sample data with missing values and outliers
    sample_data = {
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, 20, 30, np.nan, 50, 60],
        'C': ['x', 'y', 'z', 'x', 'y', 'z']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    # Clean the data
    cleaned = clean_dataset(df, strategy='mean', outlier_threshold=2)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    # Remove duplicates
    no_duplicates = remove_duplicates(cleaned)
    print("\nDataFrame without duplicates:")
    print(no_duplicates)
    
    # Normalize numeric columns
    normalized = normalize_columns(no_duplicates, method='minmax')
    print("\nNormalized DataFrame:")
    print(normalized)