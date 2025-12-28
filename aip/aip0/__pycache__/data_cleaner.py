import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=False, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_duplicates (bool): Whether to remove duplicate rows
    fill_missing (bool): Whether to fill missing values
    fill_value: Value to use for filling missing data
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing:
        missing_before = cleaned_df.isnull().sum().sum()
        cleaned_df = cleaned_df.fillna(fill_value)
        missing_after = cleaned_df.isnull().sum().sum()
        print(f"Filled {missing_before - missing_after} missing values")
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate dataset structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of columns that must be present
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_columns
        validation_results['all_required_present'] = len(missing_columns) == 0
    
    return validation_results

def sample_data_cleaning():
    """Example usage of the data cleaning functions."""
    data = {
        'id': [1, 2, 2, 3, 4, 4],
        'value': [10, 20, 20, None, 40, 40],
        'category': ['A', 'B', 'B', 'C', 'D', 'D']
    }
    
    df = pd.DataFrame(data)
    print("Original dataset:")
    print(df)
    print("\nDataset info:")
    print(df.info())
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0)
    print("\nCleaned dataset:")
    print(cleaned)
    
    validation = validate_dataset(cleaned, required_columns=['id', 'value'])
    print("\nValidation results:")
    for key, value in validation.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    sample_data_cleaning()