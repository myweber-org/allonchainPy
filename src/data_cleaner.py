
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)

def clean_dataset(df, numeric_columns=None):
    """
    Clean dataset by removing outliers from all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    numeric_columns (list): List of numeric column names. If None, uses all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
            except Exception as e:
                print(f"Warning: Could not process column '{column}': {e}")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.randint(1, 100, 1000),
        'category': np.random.choice(['X', 'Y', 'Z'], 1000)
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Original shape: {df.shape}")
    
    cleaned_df = clean_dataset(df, ['A', 'B', 'C'])
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Removed {len(df) - len(cleaned_df)} outliers")import pandas as pd
import re

def clean_dataframe(df, columns_to_check=None, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and optionally normalizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_check (list, optional): List of column names to check for duplicates.
                                          If None, uses all columns.
        normalize_text (bool): If True, normalize text in string columns by lowercasing
                              and removing extra whitespace.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    if columns_to_check is None:
        columns_to_check = cleaned_df.columns.tolist()
    
    initial_rows = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check, keep='first')
    removed_duplicates = initial_rows - len(cleaned_df)
    
    # Normalize text columns if requested
    if normalize_text:
        text_columns = cleaned_df.select_dtypes(include=['object']).columns
        for col in text_columns:
            cleaned_df[col] = cleaned_df[col].astype(str).apply(
                lambda x: re.sub(r'\s+', ' ', x.strip().lower()) if pd.notna(x) else x
            )
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    # Print cleaning summary
    print(f"Data cleaning complete:")
    print(f"  - Removed {removed_duplicates} duplicate rows")
    print(f"  - Final dataset has {len(cleaned_df)} rows and {len(cleaned_df.columns)} columns")
    
    return cleaned_df

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        email_column (str): Name of the column containing email addresses.
    
    Returns:
        pd.DataFrame: DataFrame with additional 'email_valid' boolean column.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    validated_df = df.copy()
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    validated_df['email_valid'] = validated_df[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notna(x) else False
    )
    
    valid_count = validated_df['email_valid'].sum()
    print(f"Email validation complete: {valid_count} valid emails out of {len(validated_df)}")
    
    return validated_df

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     sample_data = {
#         'name': ['John Doe', 'Jane Smith', 'John Doe', 'Bob Johnson', 'jane smith'],
#         'email': ['john@example.com', 'jane@example.com', 'john@example.com', 'bob@example', 'JANE@example.com'],
#         'age': [25, 30, 25, 35, 30]
#     }
#     
#     df = pd.DataFrame(sample_data)
#     print("Original DataFrame:")
#     print(df)
#     print()
#     
#     # Clean the data
#     cleaned = clean_dataframe(df, columns_to_check=['name', 'email'])
#     print("\nCleaned DataFrame:")
#     print(cleaned)
#     print()
#     
#     # Validate emails
#     validated = validate_email_column(cleaned, 'email')
#     print("\nDataFrame with email validation:")
#     print(validated)