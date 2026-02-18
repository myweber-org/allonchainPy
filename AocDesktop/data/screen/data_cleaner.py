
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(filepath):
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_column(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataframe(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_column(cleaned_df, col)
    return cleaned_df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    numeric_cols = ['age', 'income', 'score']
    
    try:
        raw_data = load_dataset(input_file)
        cleaned_data = clean_dataframe(raw_data, numeric_cols)
        save_cleaned_data(cleaned_data, output_file)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found")
    except Exception as e:
        print(f"An error occurred: {str(e)}")import pandas as pd
import re

def clean_dataframe(df, text_columns=None):
    """
    Clean a pandas DataFrame by removing duplicates and standardizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        text_columns (list): List of column names to apply text standardization
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    initial_rows = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    removed_duplicates = initial_rows - len(cleaned_df)
    
    # Standardize text columns if specified
    if text_columns:
        for col in text_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].apply(_standardize_text)
    
    # Remove rows with all NaN values
    cleaned_df = cleaned_df.dropna(how='all')
    
    # Log cleaning results
    print(f"Removed {removed_duplicates} duplicate rows")
    print(f"Final DataFrame shape: {cleaned_df.shape}")
    
    return cleaned_df

def _standardize_text(text):
    """
    Internal function to standardize text by:
    1. Converting to string
    2. Lowercasing
    3. Removing extra whitespace
    4. Removing special characters (keeping alphanumeric and basic punctuation)
    
    Args:
        text: Input text to standardize
    
    Returns:
        str: Standardized text
    """
    if pd.isna(text):
        return text
    
    # Convert to string
    text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters but keep basic punctuation and alphanumeric
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    return text

def validate_email_column(df, email_column):
    """
    Validate email addresses in a DataFrame column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        email_column (str): Name of the column containing email addresses
    
    Returns:
        pd.DataFrame: DataFrame with validation results
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    # Basic email regex pattern
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    # Create validation results
    validation_df = df[[email_column]].copy()
    validation_df['is_valid_email'] = validation_df[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notna(x) else False
    )
    validation_df['is_missing'] = validation_df[email_column].isna()
    
    # Calculate statistics
    valid_count = validation_df['is_valid_email'].sum()
    invalid_count = len(validation_df) - valid_count - validation_df['is_missing'].sum()
    missing_count = validation_df['is_missing'].sum()
    
    print(f"Email Validation Results:")
    print(f"  Valid emails: {valid_count}")
    print(f"  Invalid emails: {invalid_count}")
    print(f"  Missing emails: {missing_count}")
    
    return validation_df

# Example usage function
def example_usage():
    """Demonstrate how to use the data cleaning functions."""
    # Create sample data
    data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', 'Bob Johnson', None],
        'email': ['john@example.com', 'jane@example.com', 'john@example.com', 'invalid-email', None],
        'age': [25, 30, 25, 35, 40]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the DataFrame
    cleaned_df = clean_dataframe(df, text_columns=['name', 'email'])
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    # Validate email column
    validation_results = validate_email_column(cleaned_df, 'email')
    print("Email Validation Results:")
    print(validation_results)

if __name__ == "__main__":
    example_usage()