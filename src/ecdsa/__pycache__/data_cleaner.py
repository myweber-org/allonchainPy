import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None, missing_strategy='mean'):
    """
    Clean CSV data by handling missing values and standardizing columns.
    
    Args:
        file_path (str): Path to input CSV file
        output_path (str, optional): Path for cleaned output CSV
        missing_strategy (str): Strategy for handling missing values
                               ('mean', 'median', 'drop', 'zero')
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame
    """
    
    try:
        df = pd.read_csv(file_path)
        
        print(f"Original data shape: {df.shape}")
        print(f"Missing values per column:")
        print(df.isnull().sum())
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if missing_strategy == 'mean':
            for col in numeric_cols:
                df[col].fillna(df[col].mean(), inplace=True)
        elif missing_strategy == 'median':
            for col in numeric_cols:
                df[col].fillna(df[col].median(), inplace=True)
        elif missing_strategy == 'zero':
            df.fillna(0, inplace=True)
        elif missing_strategy == 'drop':
            df.dropna(inplace=True)
        
        for col in numeric_cols:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        print(f"Cleaned data shape: {df.shape}")
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pandas.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        bool: True if validation passes
    """
    if df is None or df.empty:
        print("Error: DataFrame is empty or None")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    if df.isnull().any().any():
        print("Warning: DataFrame contains missing values")
    
    return True

if __name__ == "__main__":
    sample_data = {
        'feature_a': [1, 2, np.nan, 4, 5],
        'feature_b': [10, np.nan, 30, 40, 50],
        'feature_c': [100, 200, 300, 400, 500]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', 
                               output_path='cleaned_data.csv',
                               missing_strategy='mean')
    
    if cleaned_df is not None:
        is_valid = validate_dataframe(cleaned_df, 
                                     required_columns=['feature_a', 'feature_b', 'feature_c'])
        print(f"Data validation result: {is_valid}")