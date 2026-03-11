import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): If True, remove duplicate rows.
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
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
            elif fill_missing == 'median':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    return cleaned_df

def validate_dataset(df, check_missing=True, check_types=True):
    """
    Validate a pandas DataFrame for common data quality issues.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to validate.
    check_missing (bool): If True, check for missing values.
    check_types (bool): If True, check column data types.
    
    Returns:
    dict: Dictionary containing validation results.
    """
    validation_results = {}
    
    if check_missing:
        missing_counts = df.isnull().sum()
        validation_results['missing_values'] = missing_counts[missing_counts > 0].to_dict()
        validation_results['total_missing'] = missing_counts.sum()
    
    if check_types:
        type_info = {}
        for col in df.columns:
            type_info[col] = str(df[col].dtype)
        validation_results['column_types'] = type_info
    
    validation_results['shape'] = df.shape
    validation_results['columns'] = list(df.columns)
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 30, 40, 50],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation Results:")
    print(validate_dataset(df))
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)