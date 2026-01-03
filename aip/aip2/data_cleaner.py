
import pandas as pd

def clean_dataset(df, drop_na=True, column_case='lower'):
    """
    Clean a pandas DataFrame by handling missing values and standardizing column names.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_na (bool): If True, drop rows with any null values. Default is True.
        column_case (str): Target case for column names ('lower', 'upper', or 'title'). Default is 'lower'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    # Handle missing values
    if drop_na:
        df_clean = df_clean.dropna()
    else:
        # Fill numeric columns with median, object columns with mode
        for col in df_clean.columns:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown')
    
    # Standardize column names
    if column_case == 'lower':
        df_clean.columns = df_clean.columns.str.lower()
    elif column_case == 'upper':
        df_clean.columns = df_clean.columns.str.upper()
    elif column_case == 'title':
        df_clean.columns = df_clean.columns.str.title()
    
    # Remove leading/trailing whitespace from column names
    df_clean.columns = df_clean.columns.str.strip()
    
    # Replace spaces with underscores in column names
    df_clean.columns = df_clean.columns.str.replace(' ', '_')
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and required columns.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
    Returns:
        tuple: (is_valid, message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     data = {
#         'Name': ['Alice', 'Bob', None, 'David'],
#         'Age': [25, None, 30, 35],
#         'City': ['NYC', 'LA', 'Chicago', None]
#     }
#     df = pd.DataFrame(data)
#     
#     # Clean the data
#     cleaned_df = clean_dataset(df, drop_na=False, column_case='lower')
#     print("Cleaned DataFrame:")
#     print(cleaned_df)
#     
#     # Validate the cleaned data
#     is_valid, message = validate_dataframe(cleaned_df, required_columns=['name', 'age'])
#     print(f"Validation: {is_valid} - {message}")