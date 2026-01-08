
import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to drop duplicate rows
        fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    original_shape = df.shape
    
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicates if requested
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
        duplicates_removed = original_shape[0] - cleaned_df.shape[0]
        print(f"Removed {duplicates_removed} duplicate rows")
    
    # Handle missing values
    missing_count = cleaned_df.isnull().sum().sum()
    if missing_count > 0:
        print(f"Found {missing_count} missing values")
        
        if fill_missing == 'drop':
            cleaned_df = cleaned_df.dropna()
            print(f"Dropped rows with missing values")
        elif fill_missing == 'mean':
            numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
            print(f"Filled missing numeric values with column means")
        elif fill_missing == 'median':
            numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
            print(f"Filled missing numeric values with column medians")
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    mode_value = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown'
                    cleaned_df[col] = cleaned_df[col].fillna(mode_value)
            print(f"Filled missing categorical values with column modes")
    
    # Report cleaning statistics
    final_shape = cleaned_df.shape
    print(f"Original shape: {original_shape}")
    print(f"Final shape: {final_shape}")
    print(f"Rows removed: {original_shape[0] - final_shape[0]}")
    print(f"Columns: {original_shape[1]} (unchanged)")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of column names that must be present
        min_rows (int): Minimum number of rows required
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if df.empty:
        print("Validation failed: DataFrame is empty")
        return False
    
    if len(df) < min_rows:
        print(f"Validation failed: DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if not numeric_cols.empty:
        inf_count = (df[numeric_cols] == float('inf')).sum().sum() + (df[numeric_cols] == float('-inf')).sum().sum()
        if inf_count > 0:
            print(f"Warning: Found {inf_count} infinite values in numeric columns")
    
    print("DataFrame validation passed")
    return True

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     sample_data = {
#         'id': [1, 2, 2, 3, 4, 5],
#         'value': [10.5, None, 15.2, 20.1, None, 30.4],
#         'category': ['A', 'B', 'B', None, 'C', 'A']
#     }
#     
#     df = pd.DataFrame(sample_data)
#     print("Original DataFrame:")
#     print(df)
#     print("\n" + "="*50 + "\n")
#     
#     # Clean the data
#     cleaned = clean_dataframe(df, fill_missing='mean')
#     print("\nCleaned DataFrame:")
#     print(cleaned)
#     
#     # Validate the cleaned data
#     validation_result = validate_dataframe(cleaned, required_columns=['id', 'value'], min_rows=3)
#     print(f"\nValidation result: {validation_result}")