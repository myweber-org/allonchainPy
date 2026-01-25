
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, handle_nulls='drop', null_threshold=0.5):
    """
    Clean a pandas DataFrame by handling duplicates and null values.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        drop_duplicates (bool): Whether to drop duplicate rows
        handle_nulls (str): Strategy for null handling - 'drop', 'fill_mean', 'fill_median', 'fill_mode'
        null_threshold (float): Threshold for column null percentage to drop column
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Remove duplicate rows
    if drop_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - len(df_clean)
        print(f"Removed {removed} duplicate rows")
    
    # Handle columns with excessive nulls
    null_percentages = df_clean.isnull().sum() / len(df_clean)
    cols_to_drop = null_percentages[null_percentages > null_threshold].index
    df_clean = df_clean.drop(columns=cols_to_drop)
    if len(cols_to_drop) > 0:
        print(f"Dropped {len(cols_to_drop)} columns with >{null_threshold*100}% nulls")
    
    # Handle remaining nulls based on strategy
    if handle_nulls == 'drop':
        df_clean = df_clean.dropna()
        print("Dropped rows with any null values")
    elif handle_nulls in ['fill_mean', 'fill_median', 'fill_mode']:
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            if df_clean[col].isnull().any():
                if handle_nulls == 'fill_mean':
                    fill_value = df_clean[col].mean()
                elif handle_nulls == 'fill_median':
                    fill_value = df_clean[col].median()
                else:  # fill_mode
                    fill_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 0
                df_clean[col] = df_clean[col].fillna(fill_value)
        print(f"Filled nulls using {handle_nulls} strategy")
    
    # Reset index after cleaning
    df_clean = df_clean.reset_index(drop=True)
    
    print(f"Cleaning complete. Original shape: {df.shape}, Cleaned shape: {df_clean.shape}")
    return df_clean

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
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
        print(f"Error: DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return True

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, None, 40.1, 50.0],
        'category': ['A', 'B', 'B', 'C', None, 'A'],
        'score': [100, 200, 200, None, None, 300]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned_df = clean_dataset(df, handle_nulls='fill_mean')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned data
    is_valid = validate_dataframe(cleaned_df, required_columns=['id', 'value'], min_rows=3)
    print(f"\nData validation: {'PASS' if is_valid else 'FAIL'}")