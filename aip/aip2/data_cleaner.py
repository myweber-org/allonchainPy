import pandas as pd
import numpy as np

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
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(input_file, output_file):
    df = pd.read_csv(input_file)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
    
    for col in numeric_columns:
        df = normalize_minmax(df, col)
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    print(f"Original shape: {pd.read_csv(input_file).shape}")
    print(f"Cleaned shape: {df.shape}")

if __name__ == "__main__":
    clean_dataset('raw_data.csv', 'cleaned_data.csv')
import pandas as pd
import numpy as np

def clean_dataframe(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    outlier_threshold (float): Number of standard deviations for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
    elif missing_strategy == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
    elif missing_strategy == 'mode':
        cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
    elif missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    
    # Handle outliers for numeric columns
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if cleaned_df[col].notna().any():
            mean = cleaned_df[col].mean()
            std = cleaned_df[col].std()
            
            if std > 0:  # Avoid division by zero
                z_scores = np.abs((cleaned_df[col] - mean) / std)
                outlier_mask = z_scores > outlier_threshold
                
                # Replace outliers with column median
                if outlier_mask.any():
                    median_val = cleaned_df[col].median()
                    cleaned_df.loc[outlier_mask, col] = median_val
    
    # Reset index if rows were dropped
    if missing_strategy == 'drop':
        cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
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

# Example usage
if __name__ == "__main__":
    # Create sample data with missing values and outliers
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],  # 100 is an outlier
        'B': [5, 6, 7, np.nan, 9],
        'C': ['x', 'y', 'z', 'x', 'y']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean the data
    cleaned = clean_dataframe(df, missing_strategy='median', outlier_threshold=2)
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n")
    
    # Validate the cleaned data
    is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
    print(f"Validation: {is_valid} - {message}")
import pandas as pd

def clean_dataset(df, text_columns=None, drop_threshold=0.5):
    """
    Clean a pandas DataFrame by removing null values and standardizing text columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    text_columns (list): List of column names containing text data to standardize
    drop_threshold (float): Threshold for dropping columns with too many nulls (0.0 to 1.0)
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Drop columns with too many null values
    null_ratios = cleaned_df.isnull().mean()
    columns_to_drop = null_ratios[null_ratios > drop_threshold].index.tolist()
    if columns_to_drop:
        cleaned_df = cleaned_df.drop(columns=columns_to_drop)
        print(f"Dropped columns with >{drop_threshold*100}% nulls: {columns_to_drop}")
    
    # Fill remaining null values with appropriate defaults
    for column in cleaned_df.columns:
        if cleaned_df[column].dtype == 'object':
            # For text columns, fill with empty string
            cleaned_df[column] = cleaned_df[column].fillna('')
        else:
            # For numeric columns, fill with column mean
            cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mean())
    
    # Standardize text columns if specified
    if text_columns:
        for column in text_columns:
            if column in cleaned_df.columns:
                # Convert to string, lowercase, and strip whitespace
                cleaned_df[column] = cleaned_df[column].astype(str).str.lower().str.strip()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of column names that must be present
    
    Returns:
    tuple: (is_valid, error_message)
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
#     sample_data = {
#         'name': ['Alice', 'Bob', None, 'Charlie'],
#         'age': [25, None, 30, 35],
#         'email': ['alice@test.com', 'BOB@TEST.COM', 'charlie@test.com', '']
#     }
#     
#     df = pd.DataFrame(sample_data)
#     print("Original DataFrame:")
#     print(df)
#     print("\nCleaned DataFrame:")
#     cleaned = clean_dataset(df, text_columns=['name', 'email'])
#     print(cleaned)