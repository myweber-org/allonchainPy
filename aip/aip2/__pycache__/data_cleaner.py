
def remove_duplicates(data_list):
    """
    Remove duplicate entries from a list while preserving order.
    
    Args:
        data_list: List of elements (must be hashable)
    
    Returns:
        List with duplicates removed
    """
    seen = set()
    result = []
    
    for item in data_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    
    return result

def clean_numeric_data(values, default=0):
    """
    Clean numeric data by converting strings to floats and handling invalid values.
    
    Args:
        values: List of numeric values or strings
        default: Default value for invalid entries
    
    Returns:
        List of cleaned numeric values
    """
    cleaned = []
    
    for val in values:
        try:
            cleaned.append(float(val))
        except (ValueError, TypeError):
            cleaned.append(default)
    
    return cleaned

if __name__ == "__main__":
    # Test the functions
    test_data = [1, 2, 2, 3, 4, 4, 5]
    print("Original:", test_data)
    print("Cleaned:", remove_duplicates(test_data))
    
    test_numeric = ["1.5", "2.3", "invalid", "4.7", None]
    print("Numeric original:", test_numeric)
    print("Numeric cleaned:", clean_numeric_data(test_numeric))import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (str): Method to fill missing values. Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
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
            mode_value = cleaned_df[col].mode()
            if not mode_value.empty:
                cleaned_df[col] = cleaned_df[col].fillna(mode_value.iloc[0])
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
        min_rows (int): Minimum number of rows required.
    
    Returns:
        tuple: (is_valid, message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"
import pandas as pd
import numpy as np

def clean_dataset(df, column_mapping=None, drop_duplicates=True, fill_missing=True):
    """
    Clean a pandas DataFrame by standardizing column names, removing duplicates,
    and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    column_mapping (dict): Dictionary mapping old column names to new standardized names
    drop_duplicates (bool): Whether to remove duplicate rows
    fill_missing (bool): Whether to fill missing values with appropriate defaults
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    cleaned_df = df.copy()
    
    # Standardize column names
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    # Convert column names to lowercase and replace spaces with underscores
    cleaned_df.columns = cleaned_df.columns.str.lower().str.replace(' ', '_')
    
    # Remove duplicate rows
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed_duplicates = initial_rows - len(cleaned_df)
        if removed_duplicates > 0:
            print(f"Removed {removed_duplicates} duplicate rows")
    
    # Handle missing values
    if fill_missing:
        for column in cleaned_df.columns:
            if cleaned_df[column].dtype == 'object':
                # For string columns, fill with 'unknown'
                cleaned_df[column] = cleaned_df[column].fillna('unknown')
            elif pd.api.types.is_numeric_dtype(cleaned_df[column]):
                # For numeric columns, fill with median
                cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic quality requirements.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of column names that must be present
    
    Returns:
    dict: Dictionary containing validation results and issues
    """
    
    validation_results = {
        'is_valid': True,
        'issues': [],
        'summary': {}
    }
    
    # Check for empty DataFrame
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['issues'].append('DataFrame is empty')
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f'Missing required columns: {missing_columns}')
    
    # Calculate basic statistics
    validation_results['summary']['total_rows'] = len(df)
    validation_results['summary']['total_columns'] = len(df.columns)
    validation_results['summary']['missing_values'] = df.isnull().sum().sum()
    validation_results['summary']['duplicate_rows'] = df.duplicated().sum()
    
    return validation_results

def sample_usage():
    """Demonstrate usage of the data cleaning functions."""
    
    # Create sample data
    sample_data = {
        'Customer ID': [1, 2, 3, 1, 4, None],
        'First Name': ['John', 'Jane', 'Bob', 'John', 'Alice', 'Charlie'],
        'Last Name': ['Doe', 'Smith', 'Johnson', 'Doe', 'Williams', 'Brown'],
        'Age': [25, 30, None, 25, 35, 40],
        'Purchase Amount': [100.50, 200.75, 150.00, 100.50, 300.25, 250.00]
    }
    
    df = pd.DataFrame(sample_data)
    
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Define column mapping for standardization
    column_mapping = {
        'Customer ID': 'customer_id',
        'First Name': 'first_name',
        'Last Name': 'last_name',
        'Age': 'age',
        'Purchase Amount': 'purchase_amount'
    }
    
    # Clean the dataset
    cleaned_df = clean_dataset(df, column_mapping=column_mapping)
    
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    # Validate the cleaned dataset
    required_cols = ['customer_id', 'first_name', 'last_name', 'purchase_amount']
    validation = validate_dataframe(cleaned_df, required_columns=required_cols)
    
    print("Validation Results:")
    for key, value in validation.items():
        if key != 'summary':
            print(f"{key}: {value}")
    
    print("\nSummary Statistics:")
    for key, value in validation['summary'].items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    sample_usage()