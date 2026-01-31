
import pandas as pd

def clean_dataframe(df, fill_strategy='drop', column_case='lower'):
    """
    Clean a pandas DataFrame by handling missing values and standardizing column names.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    fill_strategy (str): Strategy for handling null values. Options: 'drop', 'fill_mean', 'fill_median', 'fill_mode'.
    column_case (str): Target case for column names. Options: 'lower', 'upper', 'title'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_strategy == 'fill_mean':
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
    elif fill_strategy == 'fill_median':
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
    elif fill_strategy == 'fill_mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                mode_val = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else ''
                cleaned_df[col] = cleaned_df[col].fillna(mode_val)
    
    # Standardize column names
    if column_case == 'lower':
        cleaned_df.columns = cleaned_df.columns.str.lower()
    elif column_case == 'upper':
        cleaned_df.columns = cleaned_df.columns.str.upper()
    elif column_case == 'title':
        cleaned_df.columns = cleaned_df.columns.str.title()
    
    # Remove any leading/trailing whitespace from column names
    cleaned_df.columns = cleaned_df.columns.str.strip()
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = {
#         'Name': ['Alice', 'Bob', None, 'David'],
#         'Age': [25, None, 30, 35],
#         'Score': [85.5, 92.0, None, 78.5]
#     }
#     df = pd.DataFrame(sample_data)
#     cleaned = clean_dataframe(df, fill_strategy='fill_mean', column_case='lower')
#     print("Original DataFrame:")
#     print(df)
#     print("\nCleaned DataFrame:")
#     print(cleaned)
#     is_valid, message = validate_dataframe(cleaned, required_columns=['name', 'age'])
#     print(f"\nValidation: {message}")
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', remove_duplicates=True):
    """
    Clean a pandas DataFrame by handling missing values and removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    remove_duplicates (bool): Whether to remove duplicate rows
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    df_clean = df.copy()
    
    # Handle missing values
    if missing_strategy == 'mean':
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    elif missing_strategy == 'median':
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    elif missing_strategy == 'mode':
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown')
    elif missing_strategy == 'drop':
        df_clean = df_clean.dropna()
    
    # Remove duplicates
    if remove_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - len(df_clean)
        if removed > 0:
            print(f"Removed {removed} duplicate rows")
    
    # Reset index after cleaning
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean

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
        'warnings': [],
        'summary': {}
    }
    
    # Check if input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        validation_results['is_valid'] = False
        validation_results['errors'].append('Input is not a pandas DataFrame')
        return validation_results
    
    # Check for empty DataFrame
    if df.empty:
        validation_results['warnings'].append('DataFrame is empty')
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f'Missing required columns: {missing_columns}')
    
    # Calculate summary statistics
    validation_results['summary'] = {
        'rows': len(df),
        'columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    
    return validation_results

def normalize_numeric_columns(df, columns=None, method='minmax'):
    """
    Normalize numeric columns in DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of columns to normalize (None for all numeric)
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    
    df_normalized = df.copy()
    
    # Select columns to normalize
    if columns is None:
        numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns
    else:
        numeric_cols = [col for col in columns if col in df_normalized.columns and np.issubdtype(df_normalized[col].dtype, np.number)]
    
    # Apply normalization
    for col in numeric_cols:
        if method == 'minmax':
            min_val = df_normalized[col].min()
            max_val = df_normalized[col].max()
            if max_val > min_val:  # Avoid division by zero
                df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = df_normalized[col].mean()
            std_val = df_normalized[col].std()
            if std_val > 0:  # Avoid division by zero
                df_normalized[col] = (df_normalized[col] - mean_val) / std_val
    
    return df_normalized

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10.5, None, 15.2, 20.1, 10.5, 10.5],
        'category': ['A', 'B', None, 'A', 'B', 'B'],
        'score': [85, 92, 78, None, 88, 88]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50)
    
    # Validate data
    validation = validate_dataframe(df, required_columns=['id', 'value'])
    print("Validation Results:")
    print(validation)
    print("\n" + "="*50)
    
    # Clean data
    df_clean = clean_dataset(df, missing_strategy='mean', remove_duplicates=True)
    print("Cleaned DataFrame:")
    print(df_clean)
    print("\n" + "="*50)
    
    # Normalize numeric columns
    df_normalized = normalize_numeric_columns(df_clean, method='minmax')
    print("Normalized DataFrame:")
    print(df_normalized)