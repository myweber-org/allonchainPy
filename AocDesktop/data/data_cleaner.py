import pandas as pd

def clean_dataset(df):
    """
    Remove duplicate rows and fill missing numeric values with column mean.
    """
    # Remove duplicates
    df_cleaned = df.drop_duplicates()
    
    # Fill missing numeric values
    numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
    df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())
    
    return df_cleaned

def main():
    # Example usage
    data = {
        'A': [1, 2, None, 4, 1],
        'B': [5, None, 7, 8, 5],
        'C': ['x', 'y', 'z', 'x', 'x']
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)

if __name__ == "__main__":
    main()import csv
import os

def load_csv(file_path):
    """Load CSV file and return data as list of dictionaries."""
    data = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

def clean_numeric_fields(data, fields):
    """Remove non-numeric characters from specified fields."""
    cleaned_data = []
    for row in data:
        cleaned_row = row.copy()
        for field in fields:
            if field in cleaned_row:
                value = cleaned_row[field]
                if isinstance(value, str):
                    cleaned_value = ''.join(char for char in value if char.isdigit() or char == '.')
                    cleaned_row[field] = cleaned_value if cleaned_value else '0'
        cleaned_data.append(cleaned_row)
    return cleaned_data

def remove_empty_rows(data, required_fields):
    """Remove rows where required fields are empty."""
    return [row for row in data if all(row.get(field) for field in required_fields)]

def save_cleaned_csv(data, output_path):
    """Save cleaned data to a new CSV file."""
    if not data:
        raise ValueError("No data to save")
    
    fieldnames = data[0].keys()
    with open(output_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    return output_path

def process_csv(input_file, output_file, numeric_fields=None, required_fields=None):
    """Main function to process and clean CSV data."""
    if numeric_fields is None:
        numeric_fields = []
    if required_fields is None:
        required_fields = []
    
    try:
        data = load_csv(input_file)
        data = clean_numeric_fields(data, numeric_fields)
        data = remove_empty_rows(data, required_fields)
        save_cleaned_csv(data, output_file)
        return True, f"Data cleaned successfully. Saved to: {output_file}"
    except Exception as e:
        return False, f"Error processing file: {str(e)}"