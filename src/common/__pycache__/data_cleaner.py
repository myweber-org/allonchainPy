import pandas as pd
import numpy as np

def clean_dataset(df, strategy='mean', threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Strategy for missing value imputation ('mean', 'median', 'mode')
    threshold (float): Z-score threshold for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Handle missing values
    for column in df_clean.columns:
        if df_clean[column].dtype in ['int64', 'float64']:
            if strategy == 'mean':
                fill_value = df_clean[column].mean()
            elif strategy == 'median':
                fill_value = df_clean[column].median()
            elif strategy == 'mode':
                fill_value = df_clean[column].mode()[0]
            else:
                fill_value = 0
            
            df_clean[column] = df_clean[column].fillna(fill_value)
        else:
            df_clean[column] = df_clean[column].fillna('Unknown')
    
    # Remove outliers using Z-score method
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        z_scores = np.abs((df_clean[column] - df_clean[column].mean()) / df_clean[column].std())
        df_clean = df_clean[z_scores < threshold]
    
    return df_clean.reset_index(drop=True)

def validate_data(df, required_columns=None):
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
        'missing_columns': [],
        'empty_rows': 0,
        'duplicate_rows': 0
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_results['missing_columns'] = missing
            validation_results['is_valid'] = False
    
    validation_results['empty_rows'] = df.isnull().all(axis=1).sum()
    validation_results['duplicate_rows'] = df.duplicated().sum()
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 8],
        'C': ['x', 'y', 'z', None, 'w']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation Results:")
    print(validate_data(df, required_columns=['A', 'B', 'C']))
    
    cleaned_df = clean_dataset(df, strategy='median', threshold=2.5)
    print("\nCleaned DataFrame:")
    print(cleaned_df)