import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    original_shape = df.shape
    
    if drop_duplicates:
        df = df.drop_duplicates()
        print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows")
    
    if fill_missing:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if fill_missing == 'mean':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif fill_missing == 'median':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif fill_missing == 'zero':
            df[numeric_cols] = df[numeric_cols].fillna(0)
        
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Warning: {missing_count} missing values remain in non-numeric columns")
    
    print(f"Data cleaned: {original_shape} -> {df.shape}")
    return df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if len(df) < min_rows:
        raise ValueError(f"DataFrame has fewer than {min_rows} rows")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True

def main():
    # Example usage
    data = {
        'id': [1, 2, 2, 3, 4, 4],
        'value': [10.5, np.nan, 15.0, 20.0, np.nan, 25.0],
        'category': ['A', 'B', 'B', 'C', None, 'A']
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataframe(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    try:
        validate_dataframe(cleaned_df, required_columns=['id', 'value'], min_rows=1)
        print("\nData validation passed")
    except ValueError as e:
        print(f"\nData validation failed: {e}")

if __name__ == "__main__":
    main()import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=True):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows.
        fill_missing (bool): Whether to fill missing values with column means.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        for column in cleaned_df.select_dtypes(include=['number']).columns:
            if cleaned_df[column].isnull().any():
                cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mean())
    
    return cleaned_df

def validate_dataset(df):
    """
    Validate dataset for common issues.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
    
    Returns:
        dict: Dictionary containing validation results.
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
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
    print(validate_dataset(df))
    
    cleaned_df = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nCleaned Validation Results:")
    print(validate_dataset(cleaned_df))