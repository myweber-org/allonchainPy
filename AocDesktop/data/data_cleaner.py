import pandas as pd

def clean_dataset(df, columns_to_check=None, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    columns_to_check (list): List of column names to check for duplicates. 
                             If None, checks all columns.
    fill_missing (str): Method to fill missing values. Options: 'mean', 'median', 'mode', or 'drop'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    # Create a copy to avoid modifying the original DataFrame
    cleaned_df = df.copy()
    
    # Remove duplicates
    if columns_to_check is None:
        columns_to_check = cleaned_df.columns.tolist()
    
    cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check, keep='first')
    
    # Handle missing values
    numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing == 'mean':
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
    elif fill_missing == 'median':
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    elif fill_missing == 'mode':
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
    
    # Fill non-numeric columns with 'Unknown'
    non_numeric_cols = cleaned_df.select_dtypes(exclude=['number']).columns
    for col in non_numeric_cols:
        if cleaned_df[col].isnull().any():
            cleaned_df[col].fillna('Unknown', inplace=True)
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
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
    
    return True, "Data validation passed"

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', None, 'Eve'],
        'age': [25, 30, 30, None, 35, 40],
        'score': [85.5, 92.0, 92.0, 78.5, None, 95.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nShape:", df.shape)
    
    # Clean the data
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    print("\nShape:", cleaned.shape)
    
    # Validate the cleaned data
    is_valid, message = validate_data(cleaned, required_columns=['id', 'name', 'age', 'score'], min_rows=3)
    print(f"\nValidation: {is_valid} - {message}")import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=True):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to remove duplicate rows
        fill_missing (bool): Whether to fill missing values with column mean
    
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
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                mean_val = cleaned_df[col].mean()
                cleaned_df[col] = cleaned_df[col].fillna(mean_val)
                print(f"Filled missing values in column '{col}' with mean: {mean_val:.2f}")
    
    return cleaned_df

def validate_dataframe(df):
    """
    Validate DataFrame for common data quality issues.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
    
    Returns:
        dict: Dictionary containing validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': list(df.select_dtypes(include=['number']).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns)
    }
    
    return validation_results

def process_csv_file(input_path, output_path=None):
    """
    Process a CSV file through the data cleaning pipeline.
    
    Args:
        input_path (str): Path to input CSV file
        output_path (str, optional): Path to save cleaned CSV file
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    try:
        df = pd.read_csv(input_path)
        print(f"Loaded data from {input_path}")
        print(f"Original shape: {df.shape}")
        
        validation_before = validate_dataframe(df)
        print("\nValidation before cleaning:")
        for key, value in validation_before.items():
            print(f"{key}: {value}")
        
        cleaned_df = clean_dataframe(df)
        
        validation_after = validate_dataframe(cleaned_df)
        print("\nValidation after cleaning:")
        for key, value in validation_after.items():
            print(f"{key}: {value}")
        
        if output_path:
            cleaned_df.to_csv(output_path, index=False)
            print(f"\nCleaned data saved to {output_path}")
        
        return cleaned_df
        
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 3, 4, 5],
        'value': [10.5, 20.3, None, 30.1, 40.7, None],
        'category': ['A', 'B', 'C', 'C', 'A', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    print("Sample data created")
    cleaned = clean_dataframe(df)
    print(f"\nCleaned DataFrame:\n{cleaned}")