
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport pandas as pd

def clean_dataset(df, columns_to_check=None, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_check (list, optional): Specific columns to check for duplicates.
            If None, checks all columns. Defaults to None.
        fill_missing (str, optional): Strategy to fill missing values.
            Options: 'mean', 'median', 'mode', or 'drop'. Defaults to 'mean'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Remove duplicates
    if columns_to_check:
        cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check)
    else:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Handle missing values
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                if fill_missing == 'mean':
                    cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
                elif fill_missing == 'median':
                    cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
                elif fill_missing == 'mode':
                    cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of columns that must be present.
    
    Returns:
        tuple: (is_valid, error_message)
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
#         'A': [1, 2, 2, 4, None],
#         'B': [5, 6, 6, None, 9],
#         'C': ['x', 'y', 'y', 'z', 'x']
#     }
#     df = pd.DataFrame(data)
#     
#     # Clean the data
#     cleaned = clean_dataset(df, fill_missing='mean')
#     print(f"Original shape: {df.shape}")
#     print(f"Cleaned shape: {cleaned.shape}")
#     
#     # Validate
#     is_valid, message = validate_dataframe(cleaned, ['A', 'B'])
#     print(f"Validation: {is_valid}, Message: {message}")
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    if max_val == min_val:
        return df[column]
    return (df[column] - min_val) / (max_val - min_val)

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

def generate_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 200),
        'feature_b': np.random.exponential(50, 200),
        'feature_c': np.random.uniform(0, 1, 200)
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    sample_df = generate_sample_data()
    numeric_cols = ['feature_a', 'feature_b', 'feature_c']
    cleaned_data = clean_dataset(sample_df, numeric_cols)
    print(f"Original shape: {sample_df.shape}")
    print(f"Cleaned shape: {cleaned_data.shape}")
    print(cleaned_data.describe())
import pandas as pd

def filter_and_clean_dataframe(df, filter_column, filter_value, clean_column, clean_value=None):
    """
    Filters a DataFrame based on a column's value and cleans another column
    by replacing its values.

    Args:
        df (pd.DataFrame): The input DataFrame.
        filter_column (str): Column name to apply the filter on.
        filter_value: The value to filter by in the filter_column.
        clean_column (str): Column name to clean/modify.
        clean_value: The value to set in the clean_column for filtered rows.
                     If None, rows are dropped instead of cleaned.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    if filter_column not in df.columns or clean_column not in df.columns:
        raise ValueError("Specified columns not found in DataFrame")

    filtered_df = df[df[filter_column] == filter_value].copy()

    if clean_value is not None:
        filtered_df.loc[:, clean_column] = clean_value
        result_df = pd.concat([df[df[filter_column] != filter_value], filtered_df], ignore_index=True)
    else:
        result_df = df[df[filter_column] != filter_value].reset_index(drop=True)

    return result_df
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result