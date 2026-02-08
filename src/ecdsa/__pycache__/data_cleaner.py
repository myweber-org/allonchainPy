import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Strategy to fill missing values. 
                       Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median', 'mode']:
        for column in cleaned_df.select_dtypes(include=['number']).columns:
            if cleaned_df[column].isnull().any():
                if fill_missing == 'mean':
                    cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
                elif fill_missing == 'median':
                    cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
                elif fill_missing == 'mode':
                    cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
    
    return cleaned_df

def validate_dataframe(df):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    
    Returns:
    dict: Dictionary containing validation results.
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': [10, 11, 12, 12, 13]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation Results:")
    print(validate_dataframe(df))
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    print("\nCleaned Validation Results:")
    print(validate_dataframe(cleaned))import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None, fill_strategy='mean', drop_threshold=0.5):
    """
    Clean CSV data by handling missing values and removing low-quality columns.
    
    Parameters:
    input_path (str): Path to input CSV file
    output_path (str): Path for cleaned CSV file (optional)
    fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'zero')
    drop_threshold (float): Drop columns with missing values above this threshold (0.0-1.0)
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame
    """
    
    # Read input CSV
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    # Calculate missing percentage per column
    missing_percent = df.isnull().sum() / len(df)
    
    # Drop columns with too many missing values
    columns_to_drop = missing_percent[missing_percent > drop_threshold].index
    if len(columns_to_drop) > 0:
        print(f"Dropping columns with >{drop_threshold*100}% missing values: {list(columns_to_drop)}")
        df = df.drop(columns=columns_to_drop)
    
    # Fill remaining missing values based on strategy
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            if fill_strategy == 'mean' and pd.api.types.is_numeric_dtype(df[column]):
                fill_value = df[column].mean()
            elif fill_strategy == 'median' and pd.api.types.is_numeric_dtype(df[column]):
                fill_value = df[column].median()
            elif fill_strategy == 'mode':
                fill_value = df[column].mode()[0] if not df[column].mode().empty else np.nan
            elif fill_strategy == 'zero':
                fill_value = 0
            else:
                fill_value = df[column].ffill().bfill().iloc[0] if not df[column].ffill().bfill().empty else np.nan
            
            df[column] = df[column].fillna(fill_value)
    
    # Remove duplicate rows
    duplicates_count = df.duplicated().sum()
    if duplicates_count > 0:
        print(f"Removing {duplicates_count} duplicate rows")
        df = df.drop_duplicates()
    
    # Reset index after cleaning
    df = df.reset_index(drop=True)
    
    print(f"Cleaned data shape: {df.shape}")
    print(f"Removed {original_shape[0] - df.shape[0]} rows and {original_shape[1] - df.shape[1]} columns")
    
    # Save to output file if specified
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pandas.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check if DataFrame is empty
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append("DataFrame is empty")
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.any(np.isinf(df[col])):
            validation_results['warnings'].append(f"Column '{col}' contains infinite values")
    
    # Check data types consistency
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check for mixed types in object columns
            unique_types = set(type(x) for x in df[col].dropna())
            if len(unique_types) > 1:
                validation_results['warnings'].append(f"Column '{col}' has mixed data types: {unique_types}")
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5],
        'value': [10.5, None, 15.2, None, 20.1],
        'category': ['A', 'B', None, 'A', 'B'],
        'score': [100, 200, 300, None, 500]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_data.csv', index=False)
    
    # Clean the sample data
    cleaned_df = clean_csv_data(
        'sample_data.csv',
        output_path='cleaned_data.csv',
        fill_strategy='mean',
        drop_threshold=0.3
    )
    
    # Validate the cleaned data
    validation = validate_dataframe(cleaned_df, required_columns=['id', 'value'])
    print(f"Validation results: {validation}")