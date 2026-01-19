
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='drop', duplicate_strategy='first'):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    missing_strategy (str): Strategy for handling missing values ('drop', 'mean', 'median', 'mode')
    duplicate_strategy (str): Strategy for handling duplicates ('first', 'last', 'drop')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif missing_strategy == 'mean':
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
    elif missing_strategy == 'median':
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
    elif missing_strategy == 'mode':
        for column in cleaned_df.columns:
            cleaned_df[column].fillna(cleaned_df[column].mode()[0] if not cleaned_df[column].mode().empty else None, inplace=True)
    
    # Handle duplicates
    if duplicate_strategy == 'drop':
        cleaned_df = cleaned_df.drop_duplicates()
    elif duplicate_strategy == 'first':
        cleaned_df = cleaned_df.drop_duplicates(keep='first')
    elif duplicate_strategy == 'last':
        cleaned_df = cleaned_df.drop_duplicates(keep='last')
    
    return cleaned_df

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "Dataset is valid"

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a specific column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to check for outliers
    method (str): Method for outlier detection ('iqr' or 'zscore')
    threshold (float): Threshold for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric")
    
    data = df[column].copy()
    
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (data >= lower_bound) & (data <= upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        mask = z_scores < threshold
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return df[mask]

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'value': [10, 20, None, 40, 50, 60, 70, 80, 90, 100],
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
        'score': [100, 200, 150, 250, 300, 350, 400, 450, 500, 550]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean the dataset
    cleaned = clean_dataset(df, missing_strategy='mean', duplicate_strategy='first')
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n")
    
    # Validate the dataset
    is_valid, message = validate_dataset(cleaned, required_columns=['id', 'value', 'category'], min_rows=5)
    print(f"Validation: {is_valid} - {message}")
    print("\n")
    
    # Remove outliers
    try:
        no_outliers = remove_outliers(cleaned, 'score', method='iqr', threshold=1.5)
        print(f"DataFrame after removing outliers from 'score' column: {len(no_outliers)} rows")
    except ValueError as e:
        print(f"Error removing outliers: {e}")
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    drop_duplicates (bool): Whether to drop duplicate rows
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    return cleaned_df

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
    
    if not isinstance(df, pd.DataFrame):
        validation_results['is_valid'] = False
        validation_results['errors'].append('Input is not a pandas DataFrame')
        return validation_results
    
    if df.empty:
        validation_results['warnings'].append('DataFrame is empty')
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f'Missing required columns: {missing_cols}')
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 30, 40, 50],
        'C': ['x', 'y', 'x', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    validation = validate_dataframe(cleaned, required_columns=['A', 'B'])
    print("\nValidation Results:")
    print(validation)