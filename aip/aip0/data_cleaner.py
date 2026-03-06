
import csv
import sys

def clean_csv(input_file, output_file):
    seen = set()
    cleaned_rows = []

    try:
        with open(input_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            headers = next(reader)
            for row in reader:
                if not row:
                    continue
                row_tuple = tuple(row)
                if row_tuple in seen:
                    continue
                seen.add(row_tuple)
                cleaned_rows.append(row)

        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(headers)
            writer.writerows(cleaned_rows)

        print(f"Cleaned data saved to {output_file}")
        print(f"Removed {len(seen) - len(cleaned_rows)} duplicate rows")

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_cleaner.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    clean_csv(input_file, output_file)
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a dataset by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'drop')
    outlier_threshold (float): Z-score threshold for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean())
    elif missing_strategy == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median())
    elif missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    
    # Remove outliers using Z-score method
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
        cleaned_df = cleaned_df[z_scores < outlier_threshold]
    
    # Reset index after removing outliers
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=10):
    """
    Validate dataset structure and content.
    
    Parameters:
    df (pd.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if len(df) < min_rows:
        return False, f"Dataset has less than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Dataset is valid"

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, 20, 30, np.nan, 50, 60],
        'C': [100, 200, 300, 400, 500, 600]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\nDataset info:")
    print(df.info())
    
    # Clean the dataset
    cleaned = clean_dataset(df, missing_strategy='mean', outlier_threshold=2)
    print("\nCleaned dataset:")
    print(cleaned)
    
    # Validate the cleaned dataset
    is_valid, message = validate_data(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {is_valid} - {message}")