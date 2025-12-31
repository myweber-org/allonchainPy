import pandas as pd

def clean_dataset(df, drop_duplicates=True, fillna_strategy='mean'):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Args:
        df: pandas DataFrame to clean
        drop_duplicates: If True, remove duplicate rows
        fillna_strategy: Strategy for filling missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    # Remove duplicates if requested
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    # Handle missing values
    if fillna_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values")
    else:
        for column in cleaned_df.select_dtypes(include=['float64', 'int64']).columns:
            if cleaned_df[column].isnull().any():
                if fillna_strategy == 'mean':
                    fill_value = cleaned_df[column].mean()
                elif fillna_strategy == 'median':
                    fill_value = cleaned_df[column].median()
                elif fillna_strategy == 'mode':
                    fill_value = cleaned_df[column].mode()[0]
                else:
                    raise ValueError(f"Unsupported fillna_strategy: {fillna_strategy}")
                
                cleaned_df[column] = cleaned_df[column].fillna(fill_value)
                print(f"Filled missing values in column '{column}' with {fillna_strategy}: {fill_value}")
    
    # For categorical columns, fill with most frequent value
    for column in cleaned_df.select_dtypes(include=['object', 'category']).columns:
        if cleaned_df[column].isnull().any():
            most_frequent = cleaned_df[column].mode()[0]
            cleaned_df[column] = cleaned_df[column].fillna(most_frequent)
            print(f"Filled missing values in categorical column '{column}' with mode: '{most_frequent}'")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of column names that must be present
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'summary': {}
    }
    
    # Check if DataFrame is empty
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append("DataFrame is empty")
        return validation_results
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
    
    # Check for completely null columns
    null_columns = df.columns[df.isnull().all()].tolist()
    if null_columns:
        validation_results['warnings'].append(f"Columns with all null values: {null_columns}")
    
    # Generate summary statistics
    validation_results['summary'] = {
        'rows': len(df),
        'columns': len(df.columns),
        'null_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    
    return validation_results

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, None, 40.1, None],
        'category': ['A', 'B', 'B', 'C', None, 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Validate data
    validation = validate_dataframe(df, required_columns=['id', 'value'])
    print("Validation Results:")
    print(f"Is valid: {validation['is_valid']}")
    print(f"Errors: {validation['errors']}")
    print(f"Warnings: {validation['warnings']}")
    print(f"Summary: {validation['summary']}")
    print("\n" + "="*50 + "\n")
    
    # Clean data
    cleaned = clean_dataset(df, drop_duplicates=True, fillna_strategy='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)