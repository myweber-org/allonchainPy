import csv
import re

def clean_csv(input_file, output_file):
    """
    Clean a CSV file by removing rows with missing values,
    stripping whitespace, and standardizing date formats.
    """
    cleaned_rows = []
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        for row in reader:
            # Skip rows with any empty values
            if any(value.strip() == '' for value in row.values()):
                continue
            
            # Strip whitespace from all fields
            cleaned_row = {key: value.strip() for key, value in row.items()}
            
            # Standardize date format (MM/DD/YYYY to YYYY-MM-DD)
            date_pattern = r'(\d{1,2})/(\d{1,2})/(\d{4})'
            for key in cleaned_row:
                match = re.match(date_pattern, cleaned_row[key])
                if match:
                    month, day, year = match.groups()
                    cleaned_row[key] = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            
            cleaned_rows.append(cleaned_row)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)
    
    return len(cleaned_rows)

if __name__ == "__main__":
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    try:
        processed_count = clean_csv(input_csv, output_csv)
        print(f"Successfully processed {processed_count} rows.")
        print(f"Cleaned data saved to {output_csv}")
    except FileNotFoundError:
        print(f"Error: Input file '{input_csv}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = cleaned_df.shape[0]
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - cleaned_df.shape[0]
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                if fill_missing == 'mean':
                    fill_value = cleaned_df[col].mean()
                elif fill_missing == 'median':
                    fill_value = cleaned_df[col].median()
                elif fill_missing == 'zero':
                    fill_value = 0
                else:
                    fill_value = fill_missing
                
                cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                print(f"Filled missing values in column '{col}' with {fill_value}.")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        print("Warning: DataFrame is empty.")
    
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, np.nan, 40.1, 50.0],
        'category': ['A', 'B', 'B', 'C', 'A', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaning data...")
    
    cleaned_df = clean_dataframe(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    try:
        validate_dataframe(cleaned_df, required_columns=['id', 'value'])
        print("\nData validation passed.")
    except ValueError as e:
        print(f"\nData validation failed: {e}")