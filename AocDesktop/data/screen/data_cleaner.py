import pandas as pd
import numpy as np

def clean_csv_data(filepath, missing_strategy='mean', columns_to_drop=None):
    """
    Load and clean CSV data by handling missing values and removing specified columns.
    
    Args:
        filepath (str): Path to the CSV file.
        missing_strategy (str): Strategy for handling missing values.
            Options: 'mean', 'median', 'mode', 'drop'.
        columns_to_drop (list): List of column names to remove.
    
    Returns:
        pandas.DataFrame: Cleaned dataframe.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {filepath}")
    
    original_shape = df.shape
    
    if columns_to_drop:
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    if missing_strategy == 'drop':
        df = df.dropna()
    elif missing_strategy in ['mean', 'median']:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if missing_strategy == 'mean':
                fill_value = df[col].mean()
            else:
                fill_value = df[col].median()
            df[col] = df[col].fillna(fill_value)
    elif missing_strategy == 'mode':
        for col in df.columns:
            mode_value = df[col].mode()
            if not mode_value.empty:
                df[col] = df[col].fillna(mode_value[0])
    
    print(f"Data cleaning completed:")
    print(f"  Original shape: {original_shape}")
    print(f"  Final shape: {df.shape}")
    print(f"  Missing values handled with: {missing_strategy}")
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Args:
        df (pandas.DataFrame): Dataframe to validate.
        required_columns (list): List of required column names.
    
    Returns:
        dict: Validation results.
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append('Dataframe is empty')
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f'Missing required columns: {missing_columns}')
    
    if df.isnull().any().any():
        null_counts = df.isnull().sum()
        null_columns = null_counts[null_counts > 0].index.tolist()
        validation_results['warnings'].append(f'Columns with null values: {null_columns}')
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, np.nan, 30, 40, 50],
        'C': ['x', 'y', 'z', 'x', 'y'],
        'D': [100, 200, 300, np.nan, 500]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', missing_strategy='median', columns_to_drop=['D'])
    
    validation = validate_dataframe(cleaned_df, required_columns=['A', 'B', 'C'])
    print(f"Validation results: {validation}")
    
    import os
    os.remove('test_data.csv')