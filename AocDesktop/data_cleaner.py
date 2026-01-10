
import numpy as np

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_summary_statistics(data, column):
    mean_val = np.mean(data[column])
    median_val = np.median(data[column])
    std_val = np.std(data[column])
    return {
        'mean': mean_val,
        'median': median_val,
        'std': std_val
    }
import pandas as pd

def clean_dataset(df, columns_to_check=None, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_check (list, optional): Specific columns to check for duplicates.
                                           If None, checks all columns.
        fill_missing (str or dict): Strategy to fill missing values.
                                    Can be 'mean', 'median', 'mode', or a dictionary
                                    of column:value pairs.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    original_shape = df.shape
    
    # Remove duplicates
    if columns_to_check:
        df_cleaned = df.drop_duplicates(subset=columns_to_check)
    else:
        df_cleaned = df.drop_duplicates()
    
    duplicates_removed = original_shape[0] - df_cleaned.shape[0]
    
    # Handle missing values
    if fill_missing:
        if isinstance(fill_missing, dict):
            df_cleaned = df_cleaned.fillna(fill_missing)
        elif fill_missing == 'mean':
            df_cleaned = df_cleaned.fillna(df_cleaned.mean(numeric_only=True))
        elif fill_missing == 'median':
            df_cleaned = df_cleaned.fillna(df_cleaned.median(numeric_only=True))
        elif fill_missing == 'mode':
            for col in df_cleaned.columns:
                if df_cleaned[col].dtype == 'object':
                    mode_val = df_cleaned[col].mode()
                    if not mode_val.empty:
                        df_cleaned[col] = df_cleaned[col].fillna(mode_val.iloc[0])
    
    missing_filled = df_cleaned.isna().sum().sum()
    
    print(f"Cleaning complete:")
    print(f"  - Removed {duplicates_removed} duplicate rows")
    print(f"  - Filled {missing_filled} missing values")
    print(f"  - Original shape: {original_shape}")
    print(f"  - Cleaned shape: {df_cleaned.shape}")
    
    return df_cleaned

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of columns that must be present.
        min_rows (int): Minimum number of rows required.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Error: DataFrame is empty")
        return False
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    return True

# Example usage
if __name__ == "__main__":
    # Create sample data with duplicates and missing values
    data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', None],
        'age': [25, 30, 30, None, 35, 40],
        'score': [85.5, 92.0, 92.0, 78.5, None, 88.0]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned_df = clean_dataset(df, columns_to_check=['id', 'name'], fill_missing='mean')
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned data
    is_valid = validate_dataframe(cleaned_df, required_columns=['id', 'name', 'age', 'score'])
    print(f"\nData validation passed: {is_valid}")