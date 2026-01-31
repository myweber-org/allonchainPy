
import pandas as pd

def clean_dataset(df, remove_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by handling missing values and removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    remove_duplicates (bool): If True, remove duplicate rows.
    fill_missing (str or dict): Method to fill missing values.
                                Can be 'mean', 'median', 'mode', or a dictionary of column:value.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df.fillna(fill_missing, inplace=True)
        elif fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
        else:
            raise ValueError("Invalid fill_missing method. Use 'mean', 'median', 'mode', or a dictionary.")
    
    # Remove duplicates
    if remove_duplicates:
        cleaned_df.drop_duplicates(inplace=True)
    
    # Reset index after cleaning
    cleaned_df.reset_index(drop=True, inplace=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for required columns and data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if required_columns is not None:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    # Check for any remaining null values
    if df.isnull().any().any():
        print("Warning: Dataset contains null values after cleaning.")
    
    return True

def get_dataset_stats(df):
    """
    Get basic statistics about the dataset.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    
    Returns:
    dict: Dictionary containing dataset statistics.
    """
    stats = {
        'rows': len(df),
        'columns': len(df.columns),
        'null_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # in MB
    }
    
    # Add column-specific stats
    column_stats = {}
    for col in df.columns:
        col_type = str(df[col].dtype)
        unique_count = df[col].nunique()
        column_stats[col] = {
            'dtype': col_type,
            'unique_values': unique_count,
            'null_count': df[col].isnull().sum()
        }
    
    stats['column_details'] = column_stats
    return stats

# Example usage
if __name__ == "__main__":
    # Create sample data
    data = {
        'A': [1, 2, None, 4, 5, 5],
        'B': ['x', 'y', 'z', None, 'x', 'x'],
        'C': [10.5, 20.3, 30.1, 40.7, None, 10.5]
    }
    
    df = pd.DataFrame(data)
    print("Original dataset:")
    print(df)
    print("\nDataset statistics:")
    print(get_dataset_stats(df))
    
    # Clean the dataset
    cleaned = clean_dataset(df, remove_duplicates=True, fill_missing='mean')
    print("\nCleaned dataset:")
    print(cleaned)
    
    # Validate
    is_valid = validate_dataset(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nDataset valid: {is_valid}")
    
    print("\nCleaned dataset statistics:")
    print(get_dataset_stats(cleaned))