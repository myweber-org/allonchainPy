import pandas as pd

def clean_dataset(df, remove_duplicates=True, fill_method='drop'):
    """
    Clean a pandas DataFrame by handling missing values and optionally removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    remove_duplicates (bool): Whether to remove duplicate rows
    fill_method (str): Method to handle missing values - 'drop', 'fill_mean', 'fill_median', 'fill_mode'
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_method == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_method == 'fill_mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
    elif fill_method == 'fill_median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
    elif fill_method == 'fill_mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown')
    else:
        raise ValueError(f"Unsupported fill_method: {fill_method}")
    
    # Remove duplicates if requested
    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of column names that must be present
    
    Returns:
    dict: Dictionary with validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    # Check if input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        validation_results['is_valid'] = False
        validation_results['errors'].append('Input is not a pandas DataFrame')
        return validation_results
    
    # Check for required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f'Missing required columns: {missing_columns}')
    
    # Calculate basic statistics
    validation_results['stats']['row_count'] = len(df)
    validation_results['stats']['column_count'] = len(df.columns)
    validation_results['stats']['null_count'] = df.isnull().sum().sum()
    validation_results['stats']['duplicate_count'] = df.duplicated().sum()
    
    # Add warnings for potential issues
    if validation_results['stats']['null_count'] > 0:
        validation_results['warnings'].append(f'Found {validation_results["stats"]["null_count"]} null values')
    
    if validation_results['stats']['duplicate_count'] > 0:
        validation_results['warnings'].append(f'Found {validation_results["stats"]["duplicate_count"]} duplicate rows')
    
    if len(df) == 0:
        validation_results['warnings'].append('DataFrame is empty')
    
    return validation_results

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     sample_data = {
#         'A': [1, 2, None, 4, 1],
#         'B': [5, 6, 7, None, 5],
#         'C': ['x', 'y', 'z', 'x', 'x']
#     }
#     
#     df = pd.DataFrame(sample_data)
#     print("Original DataFrame:")
#     print(df)
#     
#     # Validate data
#     validation = validate_dataframe(df, required_columns=['A', 'B', 'C'])
#     print("\nValidation Results:")
#     print(validation)
#     
#     # Clean data
#     cleaned = clean_dataset(df, remove_duplicates=True, fill_method='fill_mean')
#     print("\nCleaned DataFrame:")
#     print(cleaned)