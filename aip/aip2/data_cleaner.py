
import pandas as pd

def clean_dataset(df, columns_to_check=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    # Create a copy to avoid modifying the original DataFrame
    df_clean = df.copy()
    
    # Remove duplicate rows
    df_clean = df_clean.drop_duplicates()
    
    # If specific columns are provided, check for missing values in those columns
    if columns_to_check:
        # Drop rows where all specified columns have missing values
        df_clean = df_clean.dropna(subset=columns_to_check, how='all')
        # For numerical columns, fill missing values with column mean
        for col in columns_to_check:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    else:
        # Drop rows where all columns are NaN
        df_clean = df_clean.dropna(how='all')
    
    # Reset index after cleaning
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean

def validate_data(df, required_columns):
    """
    Validate that the DataFrame contains all required columns.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     data = {
#         'id': [1, 2, 2, 3, 4, 5],
#         'name': ['Alice', 'Bob', 'Bob', None, 'Eve', None],
#         'age': [25, 30, 30, None, 35, 40],
#         'score': [85.5, 90.0, 90.0, 75.5, None, 88.0]
#     }
#     
#     df = pd.DataFrame(data)
#     print("Original DataFrame:")
#     print(df)
#     print("\nShape:", df.shape)
#     
#     # Clean the data
#     cleaned_df = clean_dataset(df, columns_to_check=['name', 'age', 'score'])
#     print("\nCleaned DataFrame:")
#     print(cleaned_df)
#     print("\nShape:", cleaned_df.shape)
#     
#     # Validate required columns
#     try:
#         validate_data(cleaned_df, ['id', 'name', 'age'])
#         print("\nData validation passed!")
#     except ValueError as e:
#         print(f"\nData validation failed: {e}")