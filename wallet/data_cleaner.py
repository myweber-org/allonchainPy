import pandas as pd
import numpy as np

def clean_dataset(df, column_mapping=None, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by handling missing values, renaming columns,
    and removing duplicates.
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Rename columns if mapping is provided
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    # Remove duplicate rows
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    # Handle missing values
    for column in cleaned_df.columns:
        if cleaned_df[column].isnull().any():
            missing_count = cleaned_df[column].isnull().sum()
            print(f"Column '{column}' has {missing_count} missing values")
            
            if fill_missing == 'mean' and pd.api.types.is_numeric_dtype(cleaned_df[column]):
                fill_value = cleaned_df[column].mean()
                cleaned_df[column] = cleaned_df[column].fillna(fill_value)
                print(f"  Filled with mean: {fill_value:.2f}")
            elif fill_missing == 'median' and pd.api.types.is_numeric_dtype(cleaned_df[column]):
                fill_value = cleaned_df[column].median()
                cleaned_df[column] = cleaned_df[column].fillna(fill_value)
                print(f"  Filled with median: {fill_value:.2f}")
            elif fill_missing == 'mode':
                fill_value = cleaned_df[column].mode()[0] if not cleaned_df[column].mode().empty else None
                cleaned_df[column] = cleaned_df[column].fillna(fill_value)
                print(f"  Filled with mode: {fill_value}")
            else:
                # Drop rows with missing values for non-numeric columns or if specified
                cleaned_df = cleaned_df.dropna(subset=[column])
                print(f"  Dropped rows with missing values in column '{column}'")
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    print(f"Cleaning complete. Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataframe(df, required_columns=None, numeric_columns=None):
    """
    Validate DataFrame structure and data types.
    """
    validation_results = {
        'is_valid': True,
        'issues': []
    }
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Missing required columns: {missing_columns}")
    
    # Check numeric columns
    if numeric_columns:
        non_numeric = [col for col in numeric_columns if col in df.columns 
                      and not pd.api.types.is_numeric_dtype(df[col])]
        if non_numeric:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Non-numeric columns expected to be numeric: {non_numeric}")
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.any(np.isinf(df[col])):
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Column '{col}' contains infinite values")
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 3, 4, 5],
        'value': [10.5, 20.3, None, 30.1, 40.7, None],
        'category': ['A', 'B', 'C', 'C', None, 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned = clean_dataset(df, 
                          column_mapping={'id': 'identifier', 'value': 'measurement'},
                          drop_duplicates=True,
                          fill_missing='mean')
    
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    # Validate the cleaned data
    validation = validate_dataframe(cleaned, 
                                  required_columns=['identifier', 'measurement', 'category'],
                                  numeric_columns=['measurement'])
    
    print("\nValidation Results:")
    print(f"Is valid: {validation['is_valid']}")
    if validation['issues']:
        print("Issues found:")
        for issue in validation['issues']:
            print(f"  - {issue}")