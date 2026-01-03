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
            print(f"  - {issue}")import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            except Exception as e:
                print(f"Error processing column {col}: {e}")
    
    return cleaned_df

def save_cleaned_data(df, input_path, suffix="_cleaned"):
    """
    Save cleaned DataFrame to a new CSV file.
    
    Parameters:
    df (pd.DataFrame): Cleaned DataFrame
    input_path (str): Original file path
    suffix (str): Suffix to add to filename
    
    Returns:
    str: Path to saved file
    """
    if not input_path.endswith('.csv'):
        raise ValueError("Input file must be a CSV file")
    
    output_path = input_path.replace('.csv', f'{suffix}.csv')
    df.to_csv(output_path, index=False)
    
    return output_path

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    # Add some outliers
    sample_data['A'][::100] = np.random.uniform(500, 1000, 10)
    
    df = pd.DataFrame(sample_data)
    print(f"Original data shape: {df.shape}")
    
    cleaned_df = clean_numeric_data(df)
    print(f"Cleaned data shape: {cleaned_df.shape}")
    
    # Calculate percentage of data removed
    pct_removed = ((df.shape[0] - cleaned_df.shape[0]) / df.shape[0]) * 100
    print(f"Percentage of rows removed: {pct_removed:.2f}%")