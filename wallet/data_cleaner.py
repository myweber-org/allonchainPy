import csv
import re

def clean_csv(input_file, output_file):
    """
    Clean a CSV file by removing rows with missing values,
    standardizing text fields, and converting numeric columns.
    """
    cleaned_rows = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            # Skip rows with any empty values
            if any(value.strip() == '' for value in row.values()):
                continue
            
            cleaned_row = {}
            for key, value in row.items():
                # Standardize text: trim whitespace and capitalize
                if key.lower().endswith(('name', 'title', 'category')):
                    cleaned_row[key] = value.strip().title()
                # Convert numeric fields
                elif key.lower().endswith(('price', 'amount', 'quantity')):
                    try:
                        cleaned_row[key] = float(value.replace(',', ''))
                    except ValueError:
                        continue
                # Clean general text fields
                else:
                    cleaned_row[key] = re.sub(r'\s+', ' ', value.strip())
            
            # Only add row if all fields were successfully cleaned
            if len(cleaned_row) == len(fieldnames):
                cleaned_rows.append(cleaned_row)
    
    # Write cleaned data to output file
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)
    
    return len(cleaned_rows)

def validate_email(email):
    """Validate email format using regex."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def remove_duplicates(input_file, output_file, key_columns):
    """
    Remove duplicate rows based on specified key columns.
    """
    unique_rows = []
    seen_keys = set()
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Create a tuple of key column values
            key = tuple(row[col] for col in key_columns if col in row)
            
            if key not in seen_keys:
                seen_keys.add(key)
                unique_rows.append(row)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(unique_rows)
    
    return len(unique_rows)
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column in a dataset using the IQR method.
    
    Parameters:
    data (numpy.ndarray): The dataset.
    column (int): Index of the column to process.
    
    Returns:
    numpy.ndarray: Dataset with outliers removed from the specified column.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    if column >= data.shape[1] or column < 0:
        raise IndexError("Column index out of bounds")
    
    col_data = data[:, column]
    Q1 = np.percentile(col_data, 25)
    Q3 = np.percentile(col_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    return data[mask]