
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    drop_duplicates (bool): Whether to drop duplicate rows
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    original_shape = df.shape
    
    # Remove duplicates if requested
    if drop_duplicates:
        df = df.drop_duplicates()
        print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows")
    
    # Handle missing values
    missing_count = df.isnull().sum().sum()
    
    if missing_count > 0:
        print(f"Found {missing_count} missing values")
        
        if fill_missing == 'drop':
            df = df.dropna()
            print("Dropped rows with missing values")
        elif fill_missing in ['mean', 'median']:
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                if fill_missing == 'mean':
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    df[col] = df[col].fillna(df[col].median())
            print(f"Filled missing numeric values with {fill_missing}")
        elif fill_missing == 'mode':
            for col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None)
            print("Filled missing values with mode")
    
    # Final statistics
    print(f"Original shape: {original_shape}")
    print(f"Final shape: {df.shape}")
    print(f"Remaining missing values: {df.isnull().sum().sum()}")
    
    return df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has less than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return True

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     data = {
#         'A': [1, 2, 2, None, 5],
#         'B': [10, None, 10, 40, 50],
#         'C': ['x', 'y', 'x', 'z', None]
#     }
#     
#     df = pd.DataFrame(data)
#     print("Original DataFrame:")
#     print(df)
#     
#     cleaned_df = clean_dataset(df, fill_missing='mean')
#     print("\nCleaned DataFrame:")
#     print(cleaned_df)
#     
#     is_valid = validate_dataframe(cleaned_df, required_columns=['A', 'B'])
#     print(f"\nDataFrame validation: {is_valid}")