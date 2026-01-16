
import pandas as pd

def clean_dataframe(df, drop_na=True, column_case='lower'):
    """
    Clean a pandas DataFrame by handling null values and standardizing column names.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_na (bool): If True, drop rows with any null values. Default is True.
        column_case (str): Desired case for column names. Options: 'lower', 'upper', 'title'. Default is 'lower'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_na:
        cleaned_df = cleaned_df.dropna()
    
    if column_case == 'lower':
        cleaned_df.columns = cleaned_df.columns.str.lower()
    elif column_case == 'upper':
        cleaned_df.columns = cleaned_df.columns.str.upper()
    elif column_case == 'title':
        cleaned_df.columns = cleaned_df.columns.str.title()
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a specific column using the Interquartile Range (IQR) method.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to process.
        multiplier (float): IQR multiplier for outlier detection. Default is 1.5.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed from the specified column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)
import pandas as pd

def clean_dataset(df, columns_to_check=None, fill_na_method='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_check (list, optional): List of columns to check for duplicates.
            If None, uses all columns. Defaults to None.
        fill_na_method (str, optional): Method to fill missing values.
            Options: 'mean', 'median', 'mode', or 'drop'. Defaults to 'mean'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicates
    if columns_to_check is None:
        columns_to_check = cleaned_df.columns.tolist()
    
    cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check, keep='first')
    
    # Handle missing values
    numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
    
    if fill_na_method == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_na_method == 'mean':
        for col in numeric_cols:
            if cleaned_df[col].isna().any():
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
    elif fill_na_method == 'median':
        for col in numeric_cols:
            if cleaned_df[col].isna().any():
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_na_method == 'mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].isna().any():
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of columns that must be present.
    
    Returns:
        tuple: (is_valid, message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     data = {
#         'id': [1, 2, 2, 3, 4, 5],
#         'value': [10.5, 20.3, 20.3, None, 40.1, 50.0],
#         'category': ['A', 'B', 'B', 'C', None, 'A']
#     }
#     
#     df = pd.DataFrame(data)
#     print("Original DataFrame:")
#     print(df)
#     
#     cleaned = clean_dataset(df, fill_na_method='mean')
#     print("\nCleaned DataFrame:")
#     print(cleaned)
#     
#     is_valid, message = validate_dataframe(cleaned, ['id', 'value'])
#     print(f"\nValidation: {is_valid} - {message}")