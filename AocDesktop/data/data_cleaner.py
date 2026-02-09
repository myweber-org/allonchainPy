
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    data (list or array-like): The dataset containing the column to clean.
    column (int or str): The index or name of the column to process.
    
    Returns:
    tuple: A tuple containing:
        - cleaned_data (list): Data with outliers removed.
        - outlier_indices (list): Indices of removed outliers.
    """
    if isinstance(data, list):
        data_array = np.array(data)
    else:
        data_array = data
    
    if isinstance(column, str):
        raise ValueError("Column names not supported with array input. Use integer index.")
    
    column_data = data_array[:, column].astype(float)
    
    Q1 = np.percentile(column_data, 25)
    Q3 = np.percentile(column_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_mask = (column_data < lower_bound) | (column_data > upper_bound)
    outlier_indices = np.where(outlier_mask)[0].tolist()
    
    cleaned_data = data_array[~outlier_mask].tolist()
    
    return cleaned_data, outlier_indices

def validate_data_shape(data, expected_columns):
    """
    Validate that data has the expected number of columns.
    
    Parameters:
    data (list or array-like): Data to validate.
    expected_columns (int): Expected number of columns.
    
    Returns:
    bool: True if data shape is valid, False otherwise.
    """
    if len(data) == 0:
        return True
    
    if isinstance(data, list):
        first_row = data[0]
    else:
        first_row = data[0, :]
    
    return len(first_row) == expected_columns

def example_usage():
    """
    Example demonstrating how to use the outlier removal function.
    """
    sample_data = [
        [1, 150.5],
        [2, 152.3],
        [3, 151.8],
        [4, 500.0],    # Outlier
        [5, 149.9],
        [6, 152.1],
        [7, 10.0],     # Outlier
        [8, 151.5]
    ]
    
    print("Original data:")
    for row in sample_data:
        print(f"  {row}")
    
    cleaned_data, outliers = remove_outliers_iqr(sample_data, column=1)
    
    print(f"\nRemoved {len(outliers)} outliers at indices: {outliers}")
    print("\nCleaned data:")
    for row in cleaned_data:
        print(f"  {row}")
    
    is_valid = validate_data_shape(cleaned_data, expected_columns=2)
    print(f"\nData shape valid: {is_valid}")

if __name__ == "__main__":
    example_usage()