
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to remove duplicate rows
        fill_missing (bool): Whether to fill missing values
        fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', or 'zero')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed_duplicates = initial_rows - len(cleaned_df)
        print(f"Removed {removed_duplicates} duplicate rows")
    
    # Handle missing values
    if fill_missing:
        missing_count = cleaned_df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values")
            
            for column in cleaned_df.columns:
                if cleaned_df[column].isnull().any():
                    if cleaned_df[column].dtype in ['int64', 'float64']:
                        if fill_strategy == 'mean':
                            fill_value = cleaned_df[column].mean()
                        elif fill_strategy == 'median':
                            fill_value = cleaned_df[column].median()
                        elif fill_strategy == 'zero':
                            fill_value = 0
                        else:
                            fill_value = cleaned_df[column].mode()[0] if not cleaned_df[column].mode().empty else 0
                        
                        cleaned_df[column].fillna(fill_value, inplace=True)
                        print(f"Filled missing values in '{column}' with {fill_strategy}: {fill_value}")
                    else:
                        # For non-numeric columns, use mode or empty string
                        if fill_strategy == 'mode':
                            fill_value = cleaned_df[column].mode()[0] if not cleaned_df[column].mode().empty else ''
                        else:
                            fill_value = ''
                        cleaned_df[column].fillna(fill_value, inplace=True)
                        print(f"Filled missing values in '{column}' with '{fill_value}'")
        else:
            print("No missing values found")
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate the dataset for basic integrity checks.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of columns that must be present
    
    Returns:
        dict: Dictionary with validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'column_types': df.dtypes.to_dict()
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_columns
        validation_results['all_required_columns_present'] = len(missing_columns) == 0
    
    return validation_results

# Example usage
if __name__ == "__main__":
    # Create sample data with some issues
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', None],
        'age': [25, 30, 30, None, 35, 40],
        'score': [85.5, 92.0, 92.0, 78.5, None, 88.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the dataset
    cleaned_df = clean_dataset(df, fill_strategy='mean')
    
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    # Validate the cleaned dataset
    print("\nValidation results:")
    validation = validate_dataset(cleaned_df, required_columns=['id', 'name', 'age', 'score'])
    for key, value in validation.items():
        print(f"{key}: {value}")