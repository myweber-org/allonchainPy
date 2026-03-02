
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (list, optional): Column names to consider for duplicates
        keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    removed_count = len(df) - len(cleaned_df)
    
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by removing non-numeric characters.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to clean
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce')
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of required column names
    
    Returns:
        tuple: (bool, str) Validation result and message
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame validation passed"
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: pandas DataFrame to clean
        drop_duplicates: bool, whether to remove duplicate rows
        fill_missing: str, method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        Cleaned pandas DataFrame
    """
    original_shape = df.shape
    
    if drop_duplicates:
        df = df.drop_duplicates()
        print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows")
    
    missing_before = df.isnull().sum().sum()
    
    if missing_before > 0:
        if fill_missing == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif fill_missing == 'median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif fill_missing == 'mode':
            for col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None)
        elif fill_missing == 'drop':
            df = df.dropna()
        else:
            raise ValueError("fill_missing must be 'mean', 'median', 'mode', or 'drop'")
    
    missing_after = df.isnull().sum().sum()
    print(f"Handled {missing_before - missing_after} missing values")
    print(f"Dataset shape changed from {original_shape} to {df.shape}")
    
    return df

def validate_dataset(df, check_duplicates=True, check_missing=True):
    """
    Validate the quality of a dataset.
    
    Args:
        df: pandas DataFrame to validate
        check_duplicates: bool, whether to check for duplicates
        check_missing: bool, whether to check for missing values
    
    Returns:
        dict with validation results
    """
    validation_results = {}
    
    if check_duplicates:
        duplicate_count = df.duplicated().sum()
        validation_results['duplicates'] = duplicate_count
        validation_results['has_duplicates'] = duplicate_count > 0
    
    if check_missing:
        missing_count = df.isnull().sum().sum()
        validation_results['missing_values'] = missing_count
        validation_results['has_missing'] = missing_count > 0
    
    validation_results['shape'] = df.shape
    validation_results['columns'] = list(df.columns)
    validation_results['dtypes'] = df.dtypes.to_dict()
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, 5, None],
        'B': [10, 20, 20, None, 50, 60],
        'C': ['x', 'y', 'y', 'z', None, 'x']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\nValidation results:")
    print(validate_dataset(df))
    
    cleaned_df = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned dataset:")
    print(cleaned_df)
    print("\nValidation results after cleaning:")
    print(validate_dataset(cleaned_df))