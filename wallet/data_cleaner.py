
def remove_duplicates(data_list):
    """
    Remove duplicate items from a list while preserving order.
    Returns a new list with unique elements.
    """
    seen = set()
    unique_list = []
    for item in data_list:
        if item not in seen:
            seen.add(item)
            unique_list.append(item)
    return unique_list

def clean_numeric_strings(data_list):
    """
    Convert string representations of numbers to actual integers.
    Non-numeric strings are kept as-is.
    """
    cleaned = []
    for item in data_list:
        try:
            cleaned.append(int(item))
        except (ValueError, TypeError):
            cleaned.append(item)
    return cleaned

def filter_by_type(data_list, data_type):
    """
    Filter list to only include items of specified type.
    """
    return [item for item in data_list if isinstance(item, data_type)]

def main():
    # Example usage
    sample_data = [1, 2, 2, 3, "4", "4", 5, "hello", 5]
    print("Original:", sample_data)
    
    unique_data = remove_duplicates(sample_data)
    print("Unique:", unique_data)
    
    cleaned_data = clean_numeric_strings(unique_data)
    print("Cleaned:", cleaned_data)
    
    numbers_only = filter_by_type(cleaned_data, int)
    print("Numbers only:", numbers_only)

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np

def remove_outliers(df, column, threshold=3):
    mean = df[column].mean()
    std = df[column].std()
    z_scores = np.abs((df[column] - mean) / std)
    return df[z_scores < threshold]

def normalize_column(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path, output_path):
    df = pd.read_csv(file_path)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers(df, col)
        df = normalize_column(df, col)
    
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    clean_dataset("raw_data.csv", "cleaned_data.csv")