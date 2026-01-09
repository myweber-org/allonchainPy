
import pandas as pd

def clean_dataset(df, columns_to_check=None, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    columns_to_check (list, optional): List of column names to check for duplicates.
                                       If None, checks all columns.
    fill_missing (bool): Whether to fill missing values.
    fill_value: Value to use for filling missing data.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if columns_to_check is None:
        columns_to_check = cleaned_df.columns.tolist()
    
    cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check)
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_data(df, required_columns):
    """
    Validate that DataFrame contains all required columns.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    bool: True if all required columns are present.
    """
    existing_columns = set(df.columns)
    required_set = set(required_columns)
    
    return required_set.issubset(existing_columns)

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Bob', None, 'David'],
        'age': [25, 30, 30, 35, None],
        'score': [85.5, 92.0, 92.0, 78.5, 88.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nDataFrame info:")
    print(df.info())
    
    cleaned = clean_dataset(df, columns_to_check=['id', 'name'])
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    required_cols = ['id', 'name', 'age']
    is_valid = validate_data(cleaned, required_cols)
    print(f"\nData validation result: {is_valid}")