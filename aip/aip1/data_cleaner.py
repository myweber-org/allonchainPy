
import pandas as pd
import numpy as np

def clean_dataset(df, column_mapping=None, drop_duplicates=True, fill_missing=True):
    """
    Clean a pandas DataFrame by standardizing columns, removing duplicates,
    and handling missing values.
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df.rename(columns=column_mapping, inplace=True)
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df.drop_duplicates(inplace=True)
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing:
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype in ['int64', 'float64']:
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
            elif cleaned_df[col].dtype == 'object':
                cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else '', inplace=True)
    
    cleaned_df.reset_index(drop=True, inplace=True)
    return cleaned_df

def validate_data(df, required_columns=None, numeric_columns=None):
    """
    Validate the cleaned dataset for required columns and numeric data integrity.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if numeric_columns:
        for col in numeric_columns:
            if col in df.columns:
                if df[col].dtype not in ['int64', 'float64']:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        print(f"Warning: Column {col} could not be converted to numeric")
    
    return True

def export_cleaned_data(df, output_path, format='csv'):
    """
    Export the cleaned DataFrame to a file.
    """
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'excel':
        df.to_excel(output_path, index=False)
    elif format == 'json':
        df.to_json(output_path, orient='records')
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Cleaned data exported to {output_path}")

if __name__ == "__main__":
    sample_data = {
        'Name': ['Alice', 'Bob', 'Alice', 'Charlie', None],
        'Age': [25, 30, 25, None, 35],
        'Salary': [50000, 60000, 50000, 70000, 80000],
        'Department': ['HR', 'IT', 'HR', 'Finance', 'IT']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df, column_mapping=None, drop_duplicates=True, fill_missing=True)
    print("Cleaned dataset:")
    print(cleaned)
    
    try:
        validate_data(cleaned, required_columns=['Name', 'Age', 'Salary'])
        print("\nData validation passed")
    except ValueError as e:
        print(f"\nData validation failed: {e}")import pandas as pd

def clean_dataset(df, remove_duplicates=True, fill_method='drop'):
    """
    Clean a pandas DataFrame by handling missing values and optionally removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    remove_duplicates (bool): If True, remove duplicate rows.
    fill_method (str): Method to handle missing values: 'drop' to remove rows, 
                       'ffill' to forward fill, 'bfill' to backward fill.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_method == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_method == 'ffill':
        cleaned_df = cleaned_df.ffill()
    elif fill_method == 'bfill':
        cleaned_df = cleaned_df.bfill()
    else:
        raise ValueError("fill_method must be 'drop', 'ffill', or 'bfill'")
    
    # Remove duplicates if requested
    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    dict: Dictionary with validation results.
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'null_count': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_results['missing_columns'] = missing_columns
        validation_results['all_required_columns_present'] = len(missing_columns) == 0
    
    return validation_results

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'A': [1, 2, None, 4, 5],
        'B': [5, 6, 7, None, 9],
        'C': [10, 11, 12, 13, 10]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation results:")
    print(validate_dataset(df))
    
    cleaned = clean_dataset(df, remove_duplicates=True, fill_method='ffill')
    print("\nCleaned DataFrame:")
    print(cleaned)
    print("\nValidation results after cleaning:")
    print(validate_dataset(cleaned))