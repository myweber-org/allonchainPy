import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None, missing_strategy='mean'):
    """
    Clean CSV data by handling missing values and standardizing columns.
    
    Parameters:
    file_path (str): Path to input CSV file
    output_path (str, optional): Path for cleaned output file
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'drop')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    df = pd.read_csv(file_path)
    
    print(f"Original data shape: {df.shape}")
    print(f"Missing values per column:")
    print(df.isnull().sum())
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if df[col].isnull().any():
            if missing_strategy == 'mean':
                fill_value = df[col].mean()
            elif missing_strategy == 'median':
                fill_value = df[col].median()
            elif missing_strategy == 'drop':
                df = df.dropna(subset=[col])
                continue
            else:
                raise ValueError(f"Unknown strategy: {missing_strategy}")
            
            df[col] = df[col].fillna(fill_value)
            print(f"Filled missing values in '{col}' with {missing_strategy}: {fill_value:.2f}")
    
    df = df.dropna(subset=df.select_dtypes(exclude=[np.number]).columns)
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
    
    print(f"Cleaned data shape: {df.shape}")
    print(f"Missing values after cleaning: {df.isnull().sum().sum()}")
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    dict: Validation results
    """
    
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append("DataFrame is empty")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].abs().max() > 1e6:
            validation_results['warnings'].append(f"Column '{col}' contains very large values")
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, np.nan, 30, 40, 50],
        'C': ['x', 'y', 'z', 'x', 'y']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', 'cleaned_data.csv', 'mean')
    
    validation = validate_dataframe(cleaned_df, ['A', 'B', 'C'])
    print(f"Validation results: {validation}")