
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def clean_dataset(df, numeric_columns):
    original_len = len(df)
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    removed_count = original_len - len(cleaned_df)
    print(f"Removed {removed_count} outliers from dataset")
    print(f"Original size: {original_len}, Cleaned size: {len(cleaned_df)}")
    
    return cleaned_df

def load_and_clean_data(filepath):
    try:
        df = pd.read_csv(filepath)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cleaned_df = clean_dataset(df, numeric_cols)
        return cleaned_df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    })
    
    sample_data.loc[::100, 'A'] = 500
    sample_data.loc[::50, 'B'] = 300
    
    cleaned = clean_dataset(sample_data, ['A', 'B', 'C'])
    print(f"Sample cleaned - Remaining rows: {len(cleaned)}")import pandas as pd

def clean_dataset(df, columns_to_check=None, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    columns_to_check (list, optional): List of column names to check for duplicates.
                                       If None, checks all columns.
    fill_missing (str, optional): Method to fill missing values.
                                  Options: 'mean', 'median', 'mode', or 'drop'.
                                  Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    # Create a copy to avoid modifying the original DataFrame
    cleaned_df = df.copy()
    
    # Remove duplicates
    if columns_to_check is None:
        cleaned_df = cleaned_df.drop_duplicates()
    else:
        cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check)
    
    # Handle missing values
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            elif fill_missing == 'median':
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                mode_value = cleaned_df[col].mode()
                if not mode_value.empty:
                    cleaned_df[col].fillna(mode_value[0], inplace=True)
            else:
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    
    # Reset index after cleaning
    cleaned_df.reset_index(drop=True, inplace=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate a DataFrame for basic requirements.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list, optional): List of required column names.
    min_rows (int, optional): Minimum number of rows required.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "Dataset is valid"

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     data = {
#         'A': [1, 2, 2, None, 5],
#         'B': [10, 20, 20, 40, None],
#         'C': ['x', 'y', 'y', 'z', 'x']
#     }
#     df = pd.DataFrame(data)
#     
#     # Clean the data
#     cleaned = clean_dataset(df, fill_missing='mean')
#     print("Cleaned DataFrame:")
#     print(cleaned)
#     
#     # Validate the cleaned data
#     is_valid, message = validate_dataset(cleaned, required_columns=['A', 'B'], min_rows=3)
#     print(f"\nValidation: {is_valid}")
#     print(f"Message: {message}")