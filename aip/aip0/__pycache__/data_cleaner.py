
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