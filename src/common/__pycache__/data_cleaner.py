
import csv
import re

def clean_csv(input_file, output_file, remove_duplicates=True, strip_whitespace=True):
    """
    Clean a CSV file by removing duplicates and stripping whitespace.
    
    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output cleaned CSV file.
        remove_duplicates (bool): Whether to remove duplicate rows.
        strip_whitespace (bool): Whether to strip whitespace from all fields.
    """
    rows = []
    seen = set()
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        rows.append(header)
        
        for row in reader:
            if strip_whitespace:
                row = [field.strip() if isinstance(field, str) else field for field in row]
            
            row_tuple = tuple(row)
            
            if remove_duplicates:
                if row_tuple in seen:
                    continue
                seen.add(row_tuple)
            
            rows.append(row)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(rows)

def validate_email(email):
    """
    Validate an email address using a simple regex pattern.
    
    Args:
        email (str): Email address to validate.
    
    Returns:
        bool: True if email is valid, False otherwise.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def filter_valid_emails(input_file, output_file, email_column_index):
    """
    Filter rows from a CSV file where the specified column contains a valid email.
    
    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output filtered CSV file.
        email_column_index (int): Index of the column containing email addresses.
    """
    valid_rows = []
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        valid_rows.append(header)
        
        for row in reader:
            if len(row) > email_column_index:
                email = row[email_column_index].strip()
                if validate_email(email):
                    valid_rows.append(row)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(valid_rows)

if __name__ == "__main__":
    # Example usage
    clean_csv("raw_data.csv", "cleaned_data.csv")
    filter_valid_emails("cleaned_data.csv", "valid_emails.csv", 2)import pandas as pd
import numpy as np

def clean_dataset(df, strategy='mean', outlier_threshold=3):
    """
    Clean dataset by handling missing values and removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    strategy (str): Imputation strategy ('mean', 'median', 'mode')
    outlier_threshold (float): Z-score threshold for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    
    df_clean = df.copy()
    
    # Handle missing values
    for column in df_clean.columns:
        if df_clean[column].isnull().any():
            if strategy == 'mean':
                df_clean[column].fillna(df_clean[column].mean(), inplace=True)
            elif strategy == 'median':
                df_clean[column].fillna(df_clean[column].median(), inplace=True)
            elif strategy == 'mode':
                df_clean[column].fillna(df_clean[column].mode()[0], inplace=True)
    
    # Remove outliers using Z-score method
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    z_scores = np.abs((df_clean[numeric_cols] - df_clean[numeric_cols].mean()) / 
                      df_clean[numeric_cols].std())
    
    outlier_mask = (z_scores < outlier_threshold).all(axis=1)
    df_clean = df_clean[outlier_mask].reset_index(drop=True)
    
    return df_clean

def validate_dataframe(df):
    """
    Validate dataframe structure and content.
    
    Parameters:
    df (pd.DataFrame): Dataframe to validate
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 8],
        'C': [9, 10, 11, 12, 13]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataset(df, strategy='median', outlier_threshold=2)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    validation = validate_dataframe(cleaned_df)
    print("\nValidation Results:")
    for key, value in validation.items():
        print(f"{key}: {value}")