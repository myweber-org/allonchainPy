import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: pandas DataFrame to clean
        drop_duplicates: If True, remove duplicate rows
        fill_missing: If True, fill missing values
        fill_value: Value to use for filling missing data
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of column names that must be present
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file):
    """
    Load a CSV file, clean the data by handling missing values,
    converting data types, and save the cleaned version.
    """
    try:
        df = pd.read_csv(input_file)
        
        # Display initial info
        print(f"Original data shape: {df.shape}")
        print(f"Missing values per column:\n{df.isnull().sum()}")
        
        # Fill missing numeric values with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"Filled missing values in '{col}' with median: {median_val}")
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                print(f"Filled missing values in '{col}' with mode: {mode_val}")
        
        # Convert date columns if present
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col])
                print(f"Converted '{col}' to datetime format")
            except Exception as e:
                print(f"Could not convert '{col}' to datetime: {e}")
        
        # Remove duplicate rows
        initial_rows = len(df)
        df.drop_duplicates(inplace=True)
        duplicates_removed = initial_rows - len(df)
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate rows")
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to: {output_file}")
        print(f"Final data shape: {df.shape}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {e}")
        return None

def validate_data(df):
    """
    Perform basic data validation checks.
    """
    if df is None:
        return
    
    print("\nData Validation Results:")
    print("-" * 30)
    
    # Check for remaining missing values
    missing_values = df.isnull().sum().sum()
    if missing_values == 0:
        print("✓ No missing values found")
    else:
        print(f"✗ Found {missing_values} missing values")
    
    # Check data types
    print(f"\nData Types:")
    for col in df.columns:
        dtype = df[col].dtype
        unique_count = df[col].nunique()
        print(f"  {col}: {dtype} (unique values: {unique_count})")
    
    # Basic statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\nNumeric Columns Statistics:")
        print(df[numeric_cols].describe().round(2))

if __name__ == "__main__":
    # Example usage
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_csv, output_csv)
    if cleaned_df is not None:
        validate_data(cleaned_df)